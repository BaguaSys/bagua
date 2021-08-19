#include <nccl.h>
#include <nccl_net.h>
#include <netinet/in.h>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <memory>
#include <functional>
#include <iostream>
#include <vector>

#include "bagua_net.h"

#define __hidden __attribute__((visibility("hidden")))

ncclDebugLogger_t NCCL_DEBUG_LOG;
#define NCCL_TRACE(FLAGS, ...) NCCL_DEBUG_LOG(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define NCCL_INFO(FLAGS, ...) NCCL_DEBUG_LOG(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define NCCL_WARN(...) NCCL_DEBUG_LOG(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)

void set_properties(ncclNetProperties_v4_t &lhs, NCCLNetPropertiesC &rhs)
{
    lhs.name = const_cast<char *>(rhs.name);
    lhs.pciPath = const_cast<char *>(rhs.pci_path);
    lhs.guid = rhs.guid;
    lhs.ptrSupport = rhs.ptr_support;
    lhs.speed = rhs.speed;
    lhs.port = rhs.port;
    lhs.maxComms = rhs.max_comms;
}

class BagueNet
{
public:
    static BagueNet &instance()
    {
        static BagueNet instance;
        return instance;
    }
    BagueNet(BagueNet const &) = delete;
    void operator=(BagueNet const &) = delete;

    int32_t devices(int32_t *ndev)
    {
        return bagua_net_c_devices(inner.get(), ndev);
    }

    int32_t get_properties(int32_t dev_id, ncclNetProperties_v4_t *props)
    {
        // HINT: The name and pci_path of ncclNetProperties_v4_t are both
        // references and cannot be passed in directly
        auto &inner_props = device_props.at(dev_id);
        if (!inner_props)
        {
            inner_props = std::shared_ptr<NCCLNetPropertiesC>(
                new NCCLNetPropertiesC{.name = NULL, .pci_path = NULL},
                [](NCCLNetPropertiesC *ptr)
                {
                    delete ptr->name;
                    delete ptr->pci_path;
                });
            int ret = bagua_net_c_get_properties(inner.get(), dev_id, inner_props.get());
            if (ret != 0)
            {
                return ret;
            }
        }

        set_properties(*props, *inner_props);
        return 0;
    }

    int32_t listen(int32_t dev_id, void *handle, void **listen_comm)
    {
        auto socket_listen_comm_id = std::make_unique<uintptr_t>(-1);
        int32_t ret = bagua_net_c_listen(inner.get(), dev_id, static_cast<SocketHandleC *>(handle), socket_listen_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *listen_comm = socket_listen_comm_id.release();
        return 0;
    }

    int32_t connect(int32_t dev_id, void *handle, void **send_comm)
    {
        auto socket_send_comm_id = std::make_unique<uintptr_t>(-1);
        int32_t ret = bagua_net_c_connect(inner.get(), dev_id, static_cast<SocketHandleC *>(handle), socket_send_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *send_comm = socket_send_comm_id.release();
        return 0;
    }

    int32_t accept(void *listen_comm, void **recv_comm)
    {
        uintptr_t listen_comm_id = *static_cast<uintptr_t *>(listen_comm);
        auto recv_comm_id = std::make_unique<uintptr_t>(-1);
        int32_t ret = bagua_net_c_accept(inner.get(), listen_comm_id, recv_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *recv_comm = recv_comm_id.release();
        return 0;
    }

    int32_t isend(void *send_comm, void *data, int size, void *mhandle, void **request)
    {
        uintptr_t send_comm_id = *static_cast<uintptr_t *>(send_comm);
        Buffer buf{
            .data = static_cast<uint8_t *>(data),
            .len = (uintptr_t)(size),
        };
        auto request_id = std::make_unique<uintptr_t>(-1);

        int32_t ret = bagua_net_c_isend(inner.get(), send_comm_id, buf, request_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *request = request_id.release();
        return 0;
    }

    int32_t irecv(void *recv_comm, void *data, int size, void *mhandle, void **request)
    {
        uintptr_t recv_comm_id = *static_cast<uintptr_t *>(recv_comm);
        Buffer buf{
            .data = static_cast<uint8_t *>(data),
            .len = (uintptr_t)(size),
        };
        auto request_id = std::make_unique<uintptr_t>(-1);

        int32_t ret = bagua_net_c_irecv(inner.get(), recv_comm_id, buf, request_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *request = request_id.release();
        return 0;
    }

    int32_t test(void *request, bool *done, uintptr_t *bytes)
    {
        uintptr_t request_id = *static_cast<uintptr_t *>(request);
        int32_t ret = bagua_net_c_test(inner.get(), request_id, done, bytes);
        if (ret != 0)
        {
            return ret;
        }

        return 0;
    }

    int32_t close_send(void *send_comm)
    {
        auto send_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(send_comm));
        return bagua_net_c_close_send(inner.get(), *send_comm_id);
    }

    int32_t close_recv(void *recv_comm)
    {
        auto recv_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(recv_comm));
        return bagua_net_c_close_recv(inner.get(), *recv_comm_id);
    }

    int32_t close_listen(void *listen_comm)
    {
        auto listen_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(listen_comm));
        return bagua_net_c_close_listen(inner.get(), *listen_comm_id);
    }

private:
    BagueNet()
    {
        inner = std::unique_ptr<BaguaNetC, std::function<void(BaguaNetC *)> >(
            bagua_net_c_create(),
            [](BaguaNetC *ptr)
            {
                bagua_net_c_destroy(&ptr);
            });
        int32_t ndev = -1;
        if (bagua_net_c_devices(inner.get(), &ndev) == 0 && ndev != -1)
        {
            device_props.resize(ndev);
        }
    }

private:
    std::unique_ptr<BaguaNetC, std::function<void(BaguaNetC *)> > inner;
    std::vector<std::shared_ptr<NCCLNetPropertiesC> > device_props;
};

__hidden ncclResult_t baguaNetInit(ncclDebugLogger_t logFunction)
{
    NCCL_DEBUG_LOG = logFunction;
    BagueNet::instance();

    NCCL_INFO(NCCL_ALL, "baguaNetInit!");

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetDevices(int *ndev)
{
    if (BagueNet::instance().devices((int32_t *)ndev) != 0)
    {
        NCCL_WARN("baguaNetDevices failed, ndev=%d", *ndev);
        return ncclInternalError;
    }

    NCCL_INFO(NCCL_ALL, "baguaNetDevices, ndev=%d", *ndev);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetGetProperties(int dev, ncclNetProperties_v4_t *props)
{
    int ret = BagueNet::instance().get_properties(dev, props);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetGetProperties failed, ret=%d, dev=%d", ret, dev);
        return ncclInternalError;
    }
    NCCL_INFO(NCCL_ALL, "baguaNetGetProperties, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetListen(int dev, void *handle, void **listenComm)
{
    int ret = BagueNet::instance().listen(dev, handle, listenComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetListen failed, ret=%d, dev=%d", ret, dev);
        return ncclInternalError;
    }
    NCCL_INFO(NCCL_ALL, "baguaNetListen, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetConnect(int dev, void *handle, void **sendComm)
{
    int ret = BagueNet::instance().connect(dev, handle, sendComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetConnect failed, ret=%d", ret);
        return ncclInternalError;
    }
    NCCL_INFO(NCCL_ALL, "baguaNetConnect ok, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetAccept(void *listenComm, void **recvComm)
{
    int ret = BagueNet::instance().accept(listenComm, recvComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetAccept failed, ret=%d", ret);
        return ncclInternalError;
    }
    NCCL_INFO(NCCL_ALL, "baguaNetAccept, listenComm=%p", listenComm);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetRegMr(void *comm, void *data, int size, int type, void **mhandle)
{
    NCCL_INFO(NCCL_ALL, "baguaNetRegMr, comm=%p, data=%p, type=%d", comm, data, type);
    return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

__hidden ncclResult_t baguaNetDeregMr(void *comm, void *mhandle)
{
    NCCL_TRACE(NCCL_ALL, "baguaNetDeregMr, comm=%p", comm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetIsend(void *sendComm, void *data, int size, void *mhandle, void **request)
{
    int ret = BagueNet::instance().isend(sendComm, data, size, mhandle, request);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetIsend failed, ret=%d, sendComm=%p, data=%p, size=%d", ret, sendComm, data, size);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetIsend, sendComm=%p, data=%p, size=%d, request_id=%d",
              sendComm, data, size, *(uintptr_t *)(*request));

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetIrecv(void *recvComm, void *data, int size, void *mhandle, void **request)
{
    int ret = BagueNet::instance().irecv(recvComm, data, size, mhandle, request);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetIrecv failed, ret=%d, sendComm=%p, data=%p, size=%d", ret, recvComm, data, size);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetIrecv, recvComm=%p, data=%p, size=%d, request_id=%d",
              recvComm, data, size, *(uintptr_t *)(*request));

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetFlush(void *recvComm, void *data, int size, void *mhandle, void **request)
{
    // We don't support CUDA pointers, so we don't need a flush operation
    return ncclInternalError;
}

__hidden ncclResult_t baguaNetTest(void *request, int *done, int *size)
{
    bool b_done = false;
    uintptr_t nbytes = 0;
    int ret = BagueNet::instance().test(request, &b_done, &nbytes);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetTest failed, ret=%d, request_id=%d",
                  ret, *static_cast<uintptr_t *>(request));
        return ncclInternalError;
    }
    *done = b_done ? 1 : 0;
    if (b_done && size != NULL)
    {
        *size = nbytes;
        NCCL_TRACE(NCCL_ALL, "size=%d", *size);
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetTest, request_id=%d, done=%d, nbytes=%d",
              *static_cast<uintptr_t *>(request), *done, nbytes);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseSend(void *sendComm)
{
    int ret = BagueNet::instance().close_send(sendComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseSend failed, ret=%d, sendComm=%p", ret, sendComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseSend, sendComm=%p", sendComm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseRecv(void *recvComm)
{
    int ret = BagueNet::instance().close_recv(recvComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseRecv failed, ret=%d, recvComm=%p", ret, recvComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseRecv, recvComm=%p", recvComm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseListen(void *listenComm)
{
    int ret = BagueNet::instance().close_listen(listenComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseListen failed, ret=%d, listenComm=%p", ret, listenComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseListen, listenComm=%p", listenComm);
    return ncclSuccess;
}

ncclNet_t ncclNetPlugin_v4 = {
    "BaguaNet",
    baguaNetInit,
    baguaNetDevices,
    baguaNetGetProperties,
    baguaNetListen,
    baguaNetConnect,
    baguaNetAccept,
    baguaNetRegMr,
    baguaNetDeregMr,
    baguaNetIsend,
    baguaNetIrecv,
    baguaNetFlush,
    baguaNetTest,
    baguaNetCloseSend,
    baguaNetCloseRecv,
    baguaNetCloseListen};
