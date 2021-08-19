#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <netinet/in.h>

struct BaguaNetC;

struct NCCLNetPropertiesC {
  const char *name;
  const char *pci_path;
  uint64_t guid;
  int32_t ptr_support;
  int32_t speed;
  int32_t port;
  int32_t max_comms;
};

struct SocketHandleC {
  struct sockaddr sockaddr;
};

struct Buffer {
  uint8_t *data;
  uintptr_t len;
};

extern "C" {

BaguaNetC *bagua_net_c_create();

void bagua_net_c_destroy(BaguaNetC **ptr);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_devices(BaguaNetC *ptr, int32_t *ndev);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_get_properties(BaguaNetC *ptr, int32_t dev_id, NCCLNetPropertiesC *props);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_listen(BaguaNetC *ptr,
                           int32_t dev_id,
                           SocketHandleC *socket_handle,
                           uintptr_t *socket_listen_comm_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_connect(BaguaNetC *ptr,
                            int32_t dev_id,
                            SocketHandleC *socket_handle,
                            uintptr_t *socket_send_comm_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_accept(BaguaNetC *ptr, uintptr_t listen_comm_id, uintptr_t *recv_comm_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_isend(BaguaNetC *ptr,
                          uintptr_t send_comm_id,
                          Buffer buf,
                          uintptr_t *request_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_irecv(BaguaNetC *ptr,
                          uintptr_t recv_comm_id,
                          Buffer buf,
                          uintptr_t *request_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_test(BaguaNetC *ptr, uintptr_t request_id, bool *done, uintptr_t *bytes);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_close_send(BaguaNetC *ptr, uintptr_t send_comm_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_close_recv(BaguaNetC *ptr, uintptr_t recv_comm_id);

/// Error code
/// 0: success
/// -1: null pointer
int32_t bagua_net_c_close_listen(BaguaNetC *ptr, uintptr_t listen_comm_id);

} // extern "C"
