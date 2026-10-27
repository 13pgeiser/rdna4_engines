from .radiance_node import RadianceMiniMax

# web/ holds the frontend extension: the request's size in the node's title, and the adapter
# lists fetched from the server.
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"RadianceMiniMax": RadianceMiniMax}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceMiniMax": "Radiance MiniMax (server)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
