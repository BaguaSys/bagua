/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "nccl_types.h"

#include <stdint.h>

typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  int maxComms;   // Maximum number of comms we can create
} ncclNetProperties_v3_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v3_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v3_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v3_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*flush)(void* collComm, void* data, int size, void* mhandle);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v3_t;
