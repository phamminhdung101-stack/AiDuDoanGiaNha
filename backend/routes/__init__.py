from flask import Flask
from backend.routes.public_routes import public_bp
from backend.routes.user_routes import user_bp
from backend.routes.admin_routes import admin_bp


def register_routes(app: Flask):
    """Đăng ký tất cả các View Routes vào ứng dụng Flask"""
    app.register_blueprint(public_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(admin_bp)