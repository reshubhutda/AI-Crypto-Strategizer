print("Testing imports...")

try:
    from app.api.routes import router
    print("✓ Routes imported successfully")
except Exception as e:
    print(f"✗ Routes import failed: {e}")

try:
    from app.main import app
    print("✓ Main app imported successfully")
except Exception as e:
    print(f"✗ Main app import failed: {e}")

print("Done!")