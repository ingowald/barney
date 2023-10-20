// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "barney/unstructured/qbvh.h"
#include "owl/common/math/box.h"
#include "owl/common/parallel/parallel_for.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <queue>

  namespace qbvh {
    struct BuildPrim {
      box3f bounds;
      int   primID;
    };
    inline std::ostream &operator<<(std::ostream &o, const BuildPrim &bp)
    { o << "{" << bp.primID << ":" << bp.bounds << "}"; return o; }

    /*! helper class that tracks both primitmive bounds _and_
      primitmive _centeroind_ bounds (as most builders require */
    struct BothBounds {
      box3f prim, cent;

      void extend(BothBounds &other)
      { prim.extend(other.prim); cent.extend(other.cent); }
      
      void extend(const box3f &box)
      { prim.extend(box); cent.extend(box.center()); }
    };

    inline std::ostream &operator<<(std::ostream &o, const BothBounds &bb)
    { o << "{prim=" << bb.prim << ", cent=" << bb.cent << "}"; return o; }
    


    /*! a set of 'bin's that each track the centroid and primtiive
        boudns, as well as num prims, that project into this bin. used
        during binning. note that to better facilitate parallel
        binning we often have each thread/tbb job first bin "its"
        primtiives into its own local set of bins, then only at the
        end 'push' this local one back into tha 'master' bins. */
    struct Bins {

      /*! create a new set of bins over the given 'domain' (which
          should usually be the box over the centroids of the bins we
          are supposed to be binning */
      Bins(const box3f &domain)
        : domain(domain),
          scale(vec3f(numBins)*rcp(domain.upper-domain.lower + 1e-20f))
      {}

      /*! push one singel prim (with given prim bounds box) into this set of bins */
      inline void push(const box3f &b);
      
      /*! push an entire other set of bins into this set of bins */
      inline void push(const Bins &other);

      /*! the domain we're binning in; this should usually bound all
          incoming prims' centroids to start with, but just in case
          we'll always project back into the valid range, if only to
          handle nasty cases such as flat domains, numerically
          challenging cases, etc */
      const box3f domain;
      
      /*! precomputed 'numBins/domains.span(), to save on divisions */
      const vec3f scale;

      /*! num bins to use - could eventually be a template
          parameter */
      static const int numBins = 8;
      
      /*! @{ numBins x 3 bins : 3 dimensions (x,y,z), and numBins bins
          in each dimension */
      struct {
        struct {
          /*! bounding box of all prims that project into this bin (no
              need to track centbounds here, that'll be done during
              partition */
          box3f  bounds;
          /*! num prims in this bin */
          size_t count { 0 };
        } bin[numBins];
      } dim[3];
      /* @} */
    };

    /*! push one single prim (with given prim bounds box) into this set of bins */
    inline void Bins::push(const box3f &b)
    {
      const vec3f cent = b.center();
      const vec3i bin = max(vec3i(0),min(vec3i(numBins-1),vec3i((cent-domain.lower)*scale)));
      for (int d=0;d<3;d++) {
        dim[d].bin[bin[d]].bounds.extend(b);
        dim[d].bin[bin[d]].count++;
      }
    }
    
    /*! push an entire other set of bins into this set of bins */
    inline void Bins::push(const Bins &other)
    {
      for (int d=0;d<3;d++)
        for (int b=0;b<numBins;b++) {
          dim[d].bin[b].bounds.extend(other.dim[d].bin[b].bounds);
          dim[d].bin[b].count += other.dim[d].bin[b].count;
        }
    }

      
    inline void binIt(Bins &bins,
                      const std::vector<BuildPrim> &bps,
                      size_t begin,
                      size_t end)
    {
      const size_t blockSize = 16*1024;
      if ((end-begin) > blockSize) {
        std::mutex mutex;
        parallel_for_blocked
          (begin,end,blockSize,[&](const size_t begin, const size_t end){
            Bins localBins(bins.domain);
            binIt(localBins,bps,begin,end);
            std::lock_guard<std::mutex> lock(mutex);
            bins.push(localBins);
          });
      } else {
        for (size_t i=begin;i<end;i++) {
          bins.push(bps[i].bounds);
        }
      }
    }

      
    struct SplitJob {
      SplitJob()
      {}
      SplitJob(const BothBounds &bounds,
               const size_t _begin,
               const size_t _end,
               std::vector<BuildPrim> *_src,
               std::vector<BuildPrim> *_dst)
        : bounds(bounds),
          begin(_begin),
          end(_end),
          src(_src),
          dst(_dst),
          priority(((_end-_begin)==1)?-1:area(bounds.prim))
      {
        initialized = true;
      }

      inline size_t size() { return end-begin; }

      bool initialized = false;
      BothBounds bounds;
      size_t begin=(size_t)-1, end=(size_t)-1;
      std::vector<BuildPrim> *src { nullptr };
      std::vector<BuildPrim> *dst { nullptr };
      float priority { -1.f };
    };
      
    struct SplitJobQueue {
      inline SplitJobQueue()
      {
        for (int i=0;i<(QBVH_WIDTH+1);i++)
          freeSlot[i] = &slot[i];
        numFree = QBVH_WIDTH+1;
      }
      inline size_t size() { return numActive; }
      inline SplitJob *alloc()
      {
        return freeSlot[--numFree];
      }
      inline void free(SplitJob *job)
      {
        freeSlot[numFree++] = job;
      }
      inline void insert(SplitJob *job)
      {
        activeSlot[numActive++] = job;
      }
      /*! get - but do NOT free - the given job */
      inline SplitJob *getActiveJob(int ID) { return activeSlot[ID]; }
      inline SplitJob *topAndPop() {
        assert(numActive > 0);
        int best = 0;
        for (int i=1;i<numActive;i++)
          if (activeSlot[i]->priority > activeSlot[best]->priority)
            best = i;
          
        SplitJob *bestJob = activeSlot[best];
        activeSlot[best] = activeSlot[--numActive];
        return bestJob;
      }
    private:
      SplitJob slot[QBVH_WIDTH+1];
      SplitJob *freeSlot[QBVH_WIDTH+1];
      SplitJob *activeSlot[QBVH_WIDTH+1];
      int numActive = 0;
      int numFree   = 0;
    };
      
    /*! job abstraction for splitting ONE subtree into MANY children */
    struct MultiNodeJob {
      MultiNodeJob(const size_t _allocedNodeID,
                   const BothBounds &bounds,
                   const size_t begin,
                   const size_t end,
                   std::vector<BuildPrim> *src,
                   std::vector<BuildPrim> *dst,
                   int depth
                   )
        : allocedNodeID(_allocedNodeID),
          bounds(bounds),
          begin(begin),
          end(end),
          src(src),
          dst(dst),
          depth(depth)
      {
      }

      size_t depth;
      size_t begin, end;
      size_t allocedNodeID;
      BothBounds bounds;
      std::vector<BuildPrim> *src;
      std::vector<BuildPrim> *dst;

      SplitJobQueue jobQueue;
      Node nodeToWrite;
        
      size_t size() { return end - begin; }

      void findBestSplit(SplitJob *in,
                         int                    &bestDim,
                         float                  &bestPos)
      {
        assert(in->initialized);
        bestDim = -1;
        assert(in->size() > 0);
        float bestCost = area(in->bounds.prim)*in->size();

        Bins bins(in->bounds.cent);
        assert(&in->src);
        binIt(bins,*in->src,in->begin,in->end);

        for (int d=0;d<3;d++) {
          box3f rBoundsArray[bins.numBins];
          box3f box;
          for (int i=bins.numBins-2;i>=0;--i) {
            box.extend(bins.dim[d].bin[i+1].bounds);
            rBoundsArray[i] = box;
          }
          size_t lCount = 0;
          size_t rCount = in->size();
          box3f lBounds;
          for (int i=0;i<bins.numBins-1;i++) {
            lCount += bins.dim[d].bin[i].count;
            lBounds.extend(bins.dim[d].bin[i].bounds);
            const size_t rCount = in->size()-lCount;
            const box3f  rBounds = rBoundsArray[i];
            if (!lCount || !rCount) continue;

            const float sah
              = lCount * area(lBounds)
              + rCount * area(rBounds);
            if (sah < bestCost) {
              bestCost = sah;
              bestDim  = d;
              bestPos
                = bins.domain.lower[d]
                + (bins.domain.upper[d]-bins.domain.lower[d])*(i+1)/float(bins.numBins);
            }
          }
        }
      }
        
      bool tryToPartition(SplitJob *in,
                          SplitJob **childJob)
      {
        assert(in->size() > 0);
        assert(in->initialized);
        if (in->bounds.cent.lower == in->bounds.cent.upper) {
          // std::cout << "could not split (same centroid) : " 
          //           << begin << " " << end << " " << in->bounds << std::endl;
          return false;
        }

        int bestDim;
        float bestPos;

        findBestSplit(in,bestDim,bestPos);
        if (bestDim < 0)
          return false;

        BothBounds lBounds,rBounds;
        size_t mid
          = performPartition(bestDim,bestPos,*in->src,*in->dst,in->begin,in->end,
                             lBounds,rBounds);
        if (mid == in->begin || mid == in->end) {
          // std::cout << "could not split - no gain : " << begin << " " << end << " " << mid << std::endl;
          return false;
        }

        new(childJob[0])SplitJob(lBounds,in->begin,mid,in->dst,in->src);
        assert(childJob[0]->size() > 0);
               
        new(childJob[1])SplitJob(rBounds,mid,in->end,in->dst,in->src);
        assert(childJob[1]->size() > 0);

        return true;
      }
        
      /*! perform partition with given plane, and return final
        write pos where the two sides met */
      size_t performPartition(const int dim,
                              const float pos,
                              const std::vector<BuildPrim> &src,
                              std::vector<BuildPrim> &dst,
                              const size_t all_begin, const size_t all_end,
                              BothBounds &shared_lBounds,
                              BothBounds &shared_rBounds)
      {
        const size_t blockSize = 1024;
        std::atomic<size_t> shared_lPos(all_begin);
        std::atomic<size_t> shared_rPos(all_end);
          
        std::mutex mutex;

        // std::cout << " ====================== do partition ======================" << std::endl;
        parallel_for_blocked
          (all_begin,all_end,blockSize,[&](const size_t block_begin, const size_t block_end){
            // first, count *in out block*
            size_t Nl = 0;
            for (size_t i=block_begin;i<block_end;i++)
              if (src[i].bounds.center()[dim] < pos) Nl++;
            const size_t Nr = (block_end-block_begin)-Nl;
            // second, atomically 'allocate' in the output arrays
            size_t lPos = (shared_lPos+=Nl)-Nl;
            size_t rPos = (shared_rPos-=Nr);
            // finally - write ....
            BothBounds lBounds, rBounds;
            for (size_t i=block_begin;i<block_end;i++) {
              if (src[i].bounds.center()[dim] < pos) {
                lBounds.extend(src[i].bounds);
                dst[lPos++] = src[i];
              } else {
                rBounds.extend(src[i].bounds);
                dst[rPos++] = src[i];
              }
            }
            
            std::lock_guard<std::mutex> lock(mutex);
            shared_lBounds.extend(lBounds);
            shared_rBounds.extend(rBounds);
          });
        
        assert(shared_lPos == shared_rPos);
        QBVH_DBG(std::cout << "done partitioning " << (all_end-all_begin) << "@" << std::endl;
                 std::cout << "  -> l = " << (shared_lPos-all_begin) << ":" << shared_lBounds << std::endl;
                 std::cout << "  -> r = " << (all_end-shared_lPos) << ":" << shared_rBounds << std::endl;
                 );

        return shared_lPos;
      }

      void partitionInEqualHalves(SplitJob *in,
                                  SplitJob **childJob,
                                  int numFree)
      {
        QBVH_DBG_PING;
        QBVH_DBG(for (int i=in->begin;i<in->end;i++)
                   std::cout << " " << i << ": "<< in->src[i] << std::endl);
        int Nl = in->size() / 2;
        for (int i=in->begin;i<in->end;i++)
          (*in->dst)[i] = (*in->src)[i];
        new(childJob[0]) SplitJob(in->bounds,in->begin,in->begin+Nl,in->dst,in->src);
        new(childJob[1]) SplitJob(in->bounds,in->begin+Nl,in->end,in->dst,in->src);
      }

      void inlineOrPush(SplitJob *childJob)
      {
        // TODO: inline!
        jobQueue.insert(childJob);
      }
        
      /*! priority queue of build order: 'first' is the surface
        area of the subtree, if it still needs splitting, or -1,
        if it is to become a child node */
      void execute()
      {
        nodeToWrite.initQuantization(bounds.prim);
        {
          SplitJob *job = jobQueue.alloc();
          new(job) SplitJob(bounds,begin,end,src,dst);
          jobQueue.insert(job);
          assert(job->initialized);
            
          while (jobQueue.size() < QBVH_WIDTH) {
            SplitJob *biggest_job = jobQueue.topAndPop();
            assert(biggest_job->initialized);
            if (biggest_job->priority < 0.f) {
              // won't be able to split this ... this is a LEAF!
              inlineOrPush(biggest_job);
              break;
            }
              
            SplitJob *childJob[2] = { jobQueue.alloc(),jobQueue.alloc() };
            assert(biggest_job->size() > 0);
            if (tryToPartition(biggest_job,childJob)) {
              assert(childJob[0]->size() > 0);
              assert(childJob[1]->size() > 0);
              inlineOrPush(childJob[0]);
              inlineOrPush(childJob[1]);
            } else {
              assert(biggest_job->size() >= 2);
              partitionInEqualHalves(biggest_job,childJob,QBVH_WIDTH-jobQueue.size());
              assert(childJob[0]->size() > 0);
              assert(childJob[1]->size() > 0);
              inlineOrPush(childJob[0]);
              inlineOrPush(childJob[1]);
            }
            jobQueue.free(biggest_job);
          }
        }
      }
    };
        

    struct Builder {
      Builder(BVH     &target,
              const size_t           numPrims,
              const box3f *primBounds);
        
        
      void buildInitialPrimBoundsAndWorldBounds(BothBounds &bounds,
                                                std::vector<BuildPrim> &buildPrims)
      {
        parallel_for_blocked(0,numPrims,1024,[&](const size_t begin, const size_t end){
            BothBounds blockBounds;
            std::vector<BuildPrim> blockPrims;
            for (size_t primID=begin;primID<end;primID++) {
              BuildPrim thisBP;
              thisBP.bounds = getBounds(primID);
              thisBP.primID = primID;
              if (thisBP.bounds.lower.x > thisBP.bounds.upper.x
                  ||
                  std::isnan(thisBP.bounds.lower.x))
                continue;
              blockPrims.push_back(thisBP);
              blockBounds.extend(thisBP.bounds);
            }
            std::lock_guard<std::mutex> lock(nodeArrayMutex);
            for (auto bp : blockPrims)
              buildPrims.push_back(bp);
            bounds.extend(blockBounds);
          });
      }

      size_t allocNode()
      {
        size_t thisNodeID = nextFreeNodeListSlot++;
        if (thisNodeID >= numReservedNodes) {
          std::lock_guard<std::mutex> lock(nodeArrayMutex);
          while(numReservedNodes <= thisNodeID) numReservedNodes += numReservedNodes;
          target.nodes.resize(numReservedNodes);
        }
        return thisNodeID;
        // return ;
      }
        
      void buildRec(MultiNodeJob &multiJob)
      {
        multiJob.execute();
        // serial_for(multiJob.jobQueue.size(),[&](int activeID) {
        parallel_for(multiJob.jobQueue.size(),[&](int activeID) {
            SplitJob *job = multiJob.jobQueue.getActiveJob(activeID);
            if (job->size() == 1) {
              multiJob.nodeToWrite.makeLeaf(activeID,(*job->src)[job->begin]);
            } else {
              MultiNodeJob childJob(allocNode(),job->bounds,
                                                  job->begin,job->end,
                                                  job->src,job->dst,
                                                  multiJob.depth+1);
              multiJob.nodeToWrite.makeInner(activeID,job->bounds.prim,childJob.allocedNodeID);
              buildRec(childJob);
            }                
          });
        multiJob.nodeToWrite.clearAllAfter(multiJob.jobQueue.size());
        {
          std::lock_guard<std::mutex> lock(nodeArrayMutex);
          target.nodes[multiJob.allocedNodeID] = multiJob.nodeToWrite;
        }
      }
    private:
      BVH     &target;
      const size_t           numPrims;
      const box3f *primBounds;
      
      std::mutex nodeArrayMutex;

      size_t           numReservedNodes     { 0 };
      std::atomic<int> nextFreeNodeListSlot { 0 };
    };


    Builder::Builder(BVH  &target,
              const size_t numPrims,
              const box3f *primBounds)
      : target(target),
        numPrims(numPrims)
    {
      numReservedNodes = 1024;
      target.nodes.resize(numReservedNodes);
        
      std::vector<BuildPrim> buildPrim[2];
      buildPrim[0].reserve(numPrims);

      BothBounds rootBounds;
      buildInitialPrimBoundsAndWorldBounds(rootBounds,buildPrim[0]);
      target.worldBounds = rootBounds.prim;
      buildPrim[1].resize(buildPrim[0].size());
          
      MultiNodeJob rootJob(allocNode(),rootBounds,0,buildPrim[0].size(),
                                         &buildPrim[0],&buildPrim[1],0);
          
      buildRec(rootJob);
      target.nodes.resize(nextFreeNodeListSlot);
    }

    void build(BVH     &target,
               const size_t           numPrims,
               const box3f *primBounds)
    {
      Builder(target,numPrims,primBounds);
      target.setSkipNodes();
    }


  }
