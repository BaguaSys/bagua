use nix::net::if_::InterfaceFlags;
use nix::sys::socket::{AddressFamily, InetAddr, SockAddr};
use std::fs;
use std::io;
use std::io::{Read, Write};

pub fn get_net_if_speed(device: &str) -> i32 {
    const DEFAULT_SPEED: i32 = 10000;

    let speed_path = format!("/sys/class/net/{}/speed", device);
    match fs::read_to_string(speed_path.clone()) {
        Ok(speed_str) => {
            return speed_str.trim().parse().unwrap_or(DEFAULT_SPEED);
        }
        Err(_) => {
            tracing::debug!(
                "Could not get speed from {}. Defaulting to 10 Gbps.",
                speed_path
            );
            DEFAULT_SPEED
        }
    }
}

#[derive(Debug, Clone)]
pub struct NCCLSocketDev {
    pub interface_name: String,
    pub addr: SockAddr,
    pub pci_path: String,
}

pub fn find_interfaces() -> Vec<NCCLSocketDev> {
    let nccl_socket_family = std::env::var("NCCL_SOCKET_FAMILY")
        .unwrap_or("-1".to_string())
        .parse::<i32>()
        .unwrap_or(-1);
    let nccl_socket_ifname =
        std::env::var("NCCL_SOCKET_IFNAME").unwrap_or("^docker,lo".to_string());
    // TODO @shjwudp: support parse sockaddr from NCCL_COMM_ID

    let mut search_not = Vec::<&str>::new();
    let mut search_exact = Vec::<&str>::new();
    if nccl_socket_ifname.starts_with("^") {
        search_not = nccl_socket_ifname[1..].split(",").collect();
    } else if nccl_socket_ifname.starts_with("=") {
        search_exact = nccl_socket_ifname[1..].split(",").collect();
    } else {
        search_exact = nccl_socket_ifname.split(",").collect();
    }

    let mut socket_devs = Vec::<NCCLSocketDev>::new();
    const MAX_IF_NAME_SIZE: usize = 16;
    let addrs = nix::ifaddrs::getifaddrs().unwrap();
    for ifaddr in addrs {
        match ifaddr.address {
            Some(addr) => {
                if addr.family() != AddressFamily::Inet && addr.family() != AddressFamily::Inet6 {
                    continue;
                }
                if ifaddr.flags.contains(InterfaceFlags::IFF_LOOPBACK) {
                    continue;
                }

                assert_eq!(ifaddr.interface_name.len() < MAX_IF_NAME_SIZE, true);
                let found_ifs: Vec<&NCCLSocketDev> = socket_devs
                    .iter()
                    .filter(|scoket_dev| scoket_dev.interface_name == ifaddr.interface_name)
                    .collect();
                if found_ifs.len() > 0 {
                    continue;
                }

                let pci_path = format!("/sys/class/net/{}/device", ifaddr.interface_name);
                let pci_path: String = match std::fs::canonicalize(pci_path) {
                    Ok(pci_path) => pci_path.to_str().unwrap_or("").to_string(),
                    Err(_) => "".to_string(),
                };

                socket_devs.push(NCCLSocketDev {
                    addr: addr,
                    interface_name: ifaddr.interface_name.clone(),
                    pci_path: pci_path,
                })
            }
            None => {
                tracing::warn!(
                    "interface {} with unsupported address family",
                    ifaddr.interface_name
                );
            }
        }
    }

    let search_not = &mut search_not;
    let search_exact = &mut search_exact;
    let socket_devs = socket_devs
        .iter()
        .filter({
            |socket_dev| -> bool {
                let (sockaddr, _) = socket_dev.addr.as_ffi_pair();
                if nccl_socket_family != -1 && sockaddr.sa_family as i32 != nccl_socket_family {
                    return false;
                }

                for not_interface in &*search_not {
                    if socket_dev.interface_name.starts_with(not_interface) {
                        return false;
                    }
                }
                if !(&*search_exact).is_empty() {
                    let mut ok = false;
                    for exact_interface in &*search_exact {
                        if socket_dev.interface_name.starts_with(exact_interface) {
                            ok = true;
                            break;
                        }
                    }
                    if !ok {
                        return false;
                    }
                }

                return true;
            }
        })
        .cloned()
        .collect();

    socket_devs
}

pub fn nonblocking_write_all(stream: &mut std::net::TcpStream, mut buf: &[u8]) -> io::Result<()> {
    while !buf.is_empty() {
        match stream.write(buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            Ok(n) => buf = &buf[n..],
            Err(ref e)
                if e.kind() == io::ErrorKind::Interrupted
                    || e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
        std::thread::yield_now();
    }
    Ok(())
}

pub fn nonblocking_read_exact(
    stream: &mut std::net::TcpStream,
    mut buf: &mut [u8],
) -> io::Result<()> {
    while !buf.is_empty() {
        match stream.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf;
                buf = &mut tmp[n..];
            }
            Err(ref e)
                if e.kind() == io::ErrorKind::Interrupted
                    || e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
        std::thread::yield_now();
    }
    if !buf.is_empty() {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        ))
    } else {
        Ok(())
    }
}

pub fn parse_user_pass_and_addr(raw_url: &str) -> Option<(String, String, String)> {
    let re = regex::Regex::new(r"^(?:([^:]+):([^@]+)@)?(\S+)$").unwrap();
    match re.captures(raw_url) {
        Some(caps) => {
            let username = match caps.get(1) {
                Some(username) => username.as_str().to_owned(),
                None => Default::default(),
            };
            let password = match caps.get(2) {
                Some(password) => password.as_str().to_owned(),
                None => Default::default(),
            };
            let address = caps.get(3).unwrap().as_str().to_owned();

            Some((username, password, address))
        }
        None => None,
    }
}

pub fn chunk_size(total: usize, min_chunksize: usize, expected_nchunks: usize) -> usize {
    let chunk_size = (total + expected_nchunks - 1) / expected_nchunks;
    let chunk_size = std::cmp::max(chunk_size, min_chunksize);

    chunk_size
}

/// Creates a `SockAddr` struct from libc's sockaddr.
///
/// Supports only the following address families: Unix, Inet (v4 & v6), Netlink and System.
/// Returns None for unsupported families.
///
/// # Safety
///
/// unsafe because it takes a raw pointer as argument.  The caller must
/// ensure that the pointer is valid.
#[cfg(not(target_os = "fuchsia"))]
pub(crate) unsafe fn from_libc_sockaddr(addr: *const libc::sockaddr) -> Option<SockAddr> {
    if addr.is_null() {
        None
    } else {
        match AddressFamily::from_i32(i32::from((*addr).sa_family)) {
            Some(AddressFamily::Unix) => None,
            Some(AddressFamily::Inet) => Some(SockAddr::Inet(InetAddr::V4(
                *(addr as *const libc::sockaddr_in),
            ))),
            Some(AddressFamily::Inet6) => Some(SockAddr::Inet(InetAddr::V6(
                *(addr as *const libc::sockaddr_in6),
            ))),
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Netlink) => Some(SockAddr::Netlink(
            //     NetlinkAddr(*(addr as *const libc::sockaddr_nl)))),
            // #[cfg(any(target_os = "ios", target_os = "macos"))]
            // Some(AddressFamily::System) => Some(SockAddr::SysControl(
            //     SysControlAddr(*(addr as *const libc::sockaddr_ctl)))),
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Packet) => Some(SockAddr::Link(
            //     LinkAddr(*(addr as *const libc::sockaddr_ll)))),
            // #[cfg(any(target_os = "dragonfly",
            //             target_os = "freebsd",
            //             target_os = "ios",
            //             target_os = "macos",
            //             target_os = "netbsd",
            //             target_os = "illumos",
            //             target_os = "openbsd"))]
            // Some(AddressFamily::Link) => {
            //     let ether_addr = LinkAddr(*(addr as *const libc::sockaddr_dl));
            //     if ether_addr.is_empty() {
            //         None
            //     } else {
            //         Some(SockAddr::Link(ether_addr))
            //     }
            // },
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Vsock) => Some(SockAddr::Vsock(
            //     VsockAddr(*(addr as *const libc::sockaddr_vm)))),
            // Other address families are currently not supported and simply yield a None
            // entry instead of a proper conversion to a `SockAddr`.
            Some(_) | None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nix::sys::socket::{InetAddr, IpAddr, SockAddr};

    #[test]
    fn test_parse() {
        let username = "nagle";
        let password = "1984";
        let address = "127.0.0.1:9090";

        let (user, pass, addr) =
            parse_user_pass_and_addr(&format!("{}:{}@{}", username, password, address)).unwrap();
        assert_eq!(user, username);
        assert_eq!(pass, password);
        assert_eq!(addr, address);

        let (user, pass, addr) = parse_user_pass_and_addr(address).unwrap();
        assert_eq!(user, "");
        assert_eq!(pass, "");
        assert_eq!(addr, address);
    }

    #[test]
    fn test_socket_handle() {
        let addr = InetAddr::new(IpAddr::new_v4(127, 0, 0, 1), 8123);
        let addr = SockAddr::new_inet(addr);
        let addr = unsafe {
            let (c_sockaddr, _) = addr.as_ffi_pair();
            from_libc_sockaddr(c_sockaddr).unwrap()
        };

        assert_eq!(addr.to_str(), "127.0.0.1:8123");
    }

    #[test]
    fn test_chunks() {
        let chunks = |total: usize, min_chunksize: usize, expected_nchunks: usize| -> usize {
            let size = chunk_size(total, min_chunksize, expected_nchunks);

            let mut chunk_count = total / size;
            if total % size != 0 {
                chunk_count += 1;
            }

            chunk_count
        };

        assert_eq!(chunks(1024, 1, 20), 20);
        assert_eq!(chunks(1024, 1000, 20), 2);
    }
}
