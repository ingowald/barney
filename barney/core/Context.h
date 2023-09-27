namespace barney {
  namespace core {

    struct GPU {
      RayQueue in, out;
    };

    struct GPUGroup {
      int numGPUs;
    };
    
  }
}
