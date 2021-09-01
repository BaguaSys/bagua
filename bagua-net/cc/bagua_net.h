#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <netinet/in.h>
#include <memory>
#include <vector>
#include <functional>

struct BaguaNetC;

struct NCCLNetPropertiesC
{
  const char *name;
  const char *pci_path;
  uint64_t guid;
  int32_t ptr_support;
  int32_t speed;
  int32_t port;
  int32_t max_comms;
};

struct SocketHandleC
{
  struct sockaddr sockaddr;
};

struct Buffer
{
  uint8_t *data;
  uintptr_t len;
};

extern "C"
{

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

class BaguaNet
{
public:
  static BaguaNet &instance()
  {
    static BaguaNet instance;
    return instance;
  }

  BaguaNet(BaguaNet const &) = delete;
  void operator=(BaguaNet const &) = delete;

  int32_t devices(int32_t *ndev);

  int32_t get_properties(int32_t dev_id, NCCLNetPropertiesC *props);

  int32_t listen(int32_t dev_id, void *handle, void **listen_comm);

  int32_t connect(int32_t dev_id, void *handle, void **send_comm);

  int32_t accept(void *listen_comm, void **recv_comm);

  int32_t isend(void *send_comm, void *data, int size, void *mhandle, void **request);

  int32_t irecv(void *recv_comm, void *data, int size, void *mhandle, void **request);

  int32_t test(void *request, bool *done, uintptr_t *bytes);

  int32_t close_send(void *send_comm);

  int32_t close_recv(void *recv_comm);

  int32_t close_listen(void *listen_comm);

private:
  BaguaNet();

private:
  std::unique_ptr<BaguaNetC, std::function<void(BaguaNetC *)> > inner;
  std::vector<std::shared_ptr<NCCLNetPropertiesC> > device_props;
};
