from dehaze_app.web import APP_CSS, build_interface, build_theme

demo = build_interface()
demo.launch(server_name='127.0.0.1', server_port=7862, inbrowser=False, css=APP_CSS, theme=build_theme())
