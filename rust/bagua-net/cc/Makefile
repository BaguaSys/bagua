INC:= -I. -I./v4 -I./v3
PLUGIN_SO:=libnccl-net.so
BAGUA_NET_LIB:=libbagua_net.a
NCCL_NET_V4:=./v4/nccl_net_v4.cc
NCCL_NET_V3:=./v3/nccl_net_v3.cc

default: $(PLUGIN_SO) $(BAGUA_NET_LIB)

$(PLUGIN_SO): bagua_net.cc $(NCCL_NET_V4) $(NCCL_NET_V3) $(BAGUA_NET_LIB)
	$(CXX) -v $(INC) \
	-std=c++17 -fPIC -shared \
	-o $@ $^ \
	-L. -l:$(BAGUA_NET_LIB) -lz

$(BAGUA_NET_LIB):
	cargo build --release && cp ../target/release/$(BAGUA_NET_LIB) ./$(BAGUA_NET_LIB)

test:
	cargo test --verbose

clean:
	rm -f $(PLUGIN_SO) $(BAGUA_NET_LIB)

tar: $(PLUGIN_SO)
	mkdir build && mv $(PLUGIN_SO) build \
		&& tar czf build.tar.gz build
