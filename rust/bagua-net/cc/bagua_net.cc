#include "bagua_net.h"

int32_t BaguaNet::devices(int32_t *ndev)
{
    return bagua_net_c_devices(inner.get(), ndev);
}

int32_t BaguaNet::get_properties(int32_t dev_id, NCCLNetPropertiesC *props)
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

    *props = *inner_props;
    return 0;
}

int32_t BaguaNet::listen(int32_t dev_id, void *handle, void **listen_comm)
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

int32_t BaguaNet::connect(int32_t dev_id, void *handle, void **send_comm)
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

int32_t BaguaNet::accept(void *listen_comm, void **recv_comm)
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

int32_t BaguaNet::isend(void *send_comm, void *data, int size, void *mhandle, void **request)
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

int32_t BaguaNet::irecv(void *recv_comm, void *data, int size, void *mhandle, void **request)
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

int32_t BaguaNet::test(void *request, bool *done, uintptr_t *bytes)
{
    uintptr_t request_id = *static_cast<uintptr_t *>(request);
    int32_t ret = bagua_net_c_test(inner.get(), request_id, done, bytes);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

int32_t BaguaNet::close_send(void *send_comm)
{
    auto send_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(send_comm));
    return bagua_net_c_close_send(inner.get(), *send_comm_id);
}

int32_t BaguaNet::close_recv(void *recv_comm)
{
    auto recv_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(recv_comm));
    return bagua_net_c_close_recv(inner.get(), *recv_comm_id);
}

int32_t BaguaNet::close_listen(void *listen_comm)
{
    auto listen_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(listen_comm));
    return bagua_net_c_close_listen(inner.get(), *listen_comm_id);
}

BaguaNet::BaguaNet()
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
