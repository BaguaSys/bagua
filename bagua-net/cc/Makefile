NCCL_HOME:=../../build/
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I.
PLUGIN_SO:=libnccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): bagua_net.cc
	$(CXX) $(INC) \
	-std=c++17 -fPIC -shared \
	-o $@ $^ \
	-L ../target/debug -l:libbagua_net.so

clean:
	rm -f $(PLUGIN_SO)
