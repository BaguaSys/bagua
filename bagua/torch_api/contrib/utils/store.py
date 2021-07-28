from typing import List, Dict, Optional


class Store:
    def set(self, key: str, value: str):
        pass

    def get(self, key: str) -> Optional[str]:
        pass

    def num_keys(self) -> int:
        pass

    def cleanup(self) -> bool:
        pass

    def mset(self, mapping: Dict[str, str]):
        pass

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        pass

    def status(self) -> bool:
        pass

    def shutdown(self):
        pass
