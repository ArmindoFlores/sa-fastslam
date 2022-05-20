__all__ = ["from_dir"]
import os


def from_dir(directory, prefix, extension="pkl"):
    return tuple(
        map(
            lambda f: os.path.join(directory, f),
            sorted(
                filter(
                    lambda f: f.endswith(f".{extension}") and f.startswith(prefix), 
                    os.listdir(directory)
                ),
                key=lambda f: float(f[len(prefix):-(len(extension)+1)])
            ),
        )
    )
