INC:= -I. -I./v4 -I./v3
PLUGIN_SO:=libnccl-net.so
BAGUA_NET_SO:=libbagua_net.so
NCCL_NET_V4:=./v4/nccl_net_v4.cc
NCCL_NET_V3:=./v3/nccl_net_v3.cc

default: $(PLUGIN_SO) $(BAGUA_NET_SO)

$(PLUGIN_SO): bagua_net.cc $(NCCL_NET_V4) $(NCCL_NET_V3) $(BAGUA_NET_SO)
	$(CXX) -v $(INC) \
	-std=c++17 -fPIC -shared \
	-o $@ $^ \
	-L. -l:$(BAGUA_NET_SO)

$(BAGUA_NET_SO):
	cargo build --release && cp ../target/release/$(BAGUA_NET_SO) ./$(BAGUA_NET_SO)

test:
	cargo test --verbose

clean:
	rm -f $(PLUGIN_SO) $(BAGUA_NET_SO)

tar: $(PLUGIN_SO) $(BAGUA_NET_SO)
	mkdir build && mv $(PLUGIN_SO) $(BAGUA_NET_SO) build \
		&& tar czf build.tar.gz build
