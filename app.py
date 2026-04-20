import socket

from dehaze_app.web import APP_CSS, build_interface, build_theme


def find_available_port(start_port: int = 7860, end_port: int = 7875) -> int:
    """Pick the first free localhost port in a small friendly range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No open local port was found between {start_port} and {end_port}.")


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=find_available_port(),
        inbrowser=False,
        css=APP_CSS,
        theme=build_theme(),
    )
