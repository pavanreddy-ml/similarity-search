from ise.engine import Engine, FaissEngine


class EngineRegistry:
    __registry = {
        "faiss": FaissEngine
    }

    @classmethod
    def get_engine(cls, engine: str) -> Engine:
        if isinstance(engine, Engine):
            return engine
        
        if engine not in cls.__registry:
            raise ValueError(f"Engine {engine} not found in the registry.")
        return cls.__registry[engine]