import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["TZ"] = "Etc/UTC"
test_root = tempfile.mkdtemp(prefix="smartjoin-astrbot-tests-")
os.environ["ASTRBOT_ROOT"] = test_root
atexit.register(shutil.rmtree, test_root, ignore_errors=True)

package_root = Path(__file__).resolve().parents[2]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))
