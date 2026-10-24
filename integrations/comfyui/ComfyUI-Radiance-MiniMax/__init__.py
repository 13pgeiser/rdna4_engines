from .radiance_node import RadianceMiniMax

# web/ carries the frontend extension that mirrors the tier math into the node
# title ("480p 16:9 -> 864x480"); the server-side Python stays the authority.
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"RadianceMiniMax": RadianceMiniMax}
NODE_DISPLAY_NAME_MAPPINGS = {"RadianceMiniMax": "Radiance MiniMax (server)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
