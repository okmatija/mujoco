import subprocess
import time
import platform
import sys
import os

debug_glTexImage2D_commands = False

# Cross-platform single character input without waiting for Enter
if platform.system() == "Windows":
    import msvcrt

    def getch():
        return msvcrt.getch().decode('utf-8')
else:
    import tty
    import termios

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def get_choice():
    """Reads a single valid character choice without waiting for Enter."""
    prompt = (
        "\nDid it work?\n"
        "(p)ass and try next scene. (P)ass and skip next scenes\n"
        "(f)ail and try next scene. (F)ail and skip next scenes\n"
        "(q)uit.\n"
    )
    print(prompt, end="", flush=True)
    while True:
        ch = getch()
        if ch in "pPfFq":
            print(ch)  # Echo the chosen character
            return ch
        # Ignore any other characters

# TODO Collect a note when the run starts
# TODO Remove requirement to press Return

configs = [
    {"internal": "XXXX",  "format": "XXXX", "type": "XXXX"},
    # {"internal": "GL_DEPTH_COMPONENT32F", "format": "GL_DEPTH_COMPONENT", "type": "GL_FLOAT"},
    # {"internal": "GL_DEPTH_COMPONENT16",  "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_SHORT"},
]

scenes = [
    ["../../mujoco_tests/mujoco_menagerie/apptronik_apollo/scene.xml", "apptronik_apollo"],
    ["../../mujoco_tests/mujoco_menagerie/ufactory_lite6/scene.xml", "ufactory_lite6"],
    ["../../mujoco_tests/mink/examples/kuka_iiwa_14/scene.xml", "kuka_iiwa_14"],
    ["../../mujoco_tests/mink/examples/ufactory_xarm7/scene.xml", "ufactory_xarm7"],
    ["../../mujoco_tests/mujoco_menagerie/flybody/scene.xml", "flybody"],

    # This scene doesn't load: "Error: mesh volume is too small: base_link_2 . Try setting inertia to shell"
    # ["../../mujoco_tests/mink/examples/hello_robot_stretch_3/scene.xml", "hello_robot_stretch_3"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_g1/scene.xml", "unitree_g1"],  # FIXME: Lights does not move with robot?
    ["../../mujoco_tests/mujoco_menagerie/umi_gripper/scene.xml", "umi_gripper"],  # NOTE: The lights don't create shadows in the scene even on linux nvidia
    ["../../mujoco_tests/mujoco_menagerie/shadow_dexee/scene.xml", "shadow_dexee"],  # NOTE: mujoco.pid is not found.
    
    ["../../mujoco_tests/mink/examples/stanford_tidybot/scene.xml", "stanford_tidybot"],
    ["../../mujoco_tests/mink/examples/unitree_h1/scene.xml", "unitree_h1"],
    ["../../mujoco_tests/mink/examples/apptronik_apollo/scene.xml", "apptronik_apollo"],  # Note: Robot falls through the floor
    ["../../mujoco_tests/mink/examples/unitree_go1/scene.xml", "unitree_go1"],
    ["../../mujoco_tests/mink/examples/universal_robots_ur5e/scene.xml", "universal_robots_ur5e"],
    ["../../mujoco_tests/mink/examples/boston_dynamics_spot/scene.xml", "boston_dynamics_spot"],
    ["../../mujoco_tests/mink/examples/aloha/scene.xml", "aloha"],
    ["../../mujoco_tests/mink/examples/unitree_g1/scene.xml", "unitree_g1"],
    
    ["../../mujoco_tests/mujoco_menagerie/franka_emika_panda/scene.xml", "franka_emika_panda"],
    ["../../mujoco_tests/mujoco_menagerie/agility_cassie/scene.xml", "agility_cassie"],
    ["../../mujoco_tests/mujoco_menagerie/agilex_piper/scene.xml", "agilex_piper"],
    ["../../mujoco_tests/mujoco_menagerie/aloha/scene.xml", "aloha"],
    ["../../mujoco_tests/mujoco_menagerie/anybotics_anymal_b/scene.xml", "anybotics_anymal_b"],
    ["../../mujoco_tests/mujoco_menagerie/anybotics_anymal_c/scene.xml", "anybotics_anymal_c"],
    ["../../mujoco_tests/mujoco_menagerie/berkeley_humanoid/scene.xml", "berkeley_humanoid"],
    ["../../mujoco_tests/mujoco_menagerie/bitcraze_crazyflie_2/scene.xml", "bitcraze_crazyflie_2"],
    ["../../mujoco_tests/mujoco_menagerie/booster_t1/scene.xml", "booster_t1"],
    ["../../mujoco_tests/mujoco_menagerie/boston_dynamics_spot/scene.xml", "boston_dynamics_spot"],
    ["../../mujoco_tests/mujoco_menagerie/franka_fr3/scene.xml", "franka_fr3"],
    ["../../mujoco_tests/mujoco_menagerie/google_barkour_v0/scene.xml", "google_barkour_v0"],
    ["../../mujoco_tests/mujoco_menagerie/google_barkour_vb/scene.xml", "google_barkour_vb"],
    ["../../mujoco_tests/mujoco_menagerie/google_robot/scene.xml", "google_robot"],
    ["../../mujoco_tests/mujoco_menagerie/hello_robot_stretch/scene.xml", "hello_robot_stretch"],
    ["../../mujoco_tests/mujoco_menagerie/hello_robot_stretch_3/scene.xml", "hello_robot_stretch_3"],
    ["../../mujoco_tests/mujoco_menagerie/kinova_gen3/scene.xml", "kinova_gen3"],
    ["../../mujoco_tests/mujoco_menagerie/kuka_iiwa_14/scene.xml", "kuka_iiwa_14"],
    ["../../mujoco_tests/mujoco_menagerie/rethink_robotics_sawyer/scene.xml", "rethink_robotics_sawyer"],
    ["../../mujoco_tests/mujoco_menagerie/robotiq_2f85/scene.xml", "robotiq_2f85"],
    ["../../mujoco_tests/mujoco_menagerie/robotiq_2f85_v4/scene.xml", "robotiq_2f85_v4"],
    ["../../mujoco_tests/mujoco_menagerie/robotis_op3/scene.xml", "robotis_op3"],
    ["../../mujoco_tests/mujoco_menagerie/skydio_x2/scene.xml", "skydio_x2"],
    ["../../mujoco_tests/mujoco_menagerie/stanford_tidybot/scene.xml", "stanford_tidybot"],
    ["../../mujoco_tests/mujoco_menagerie/trossen_vx300s/scene.xml", "trossen_vx300s"],
    ["../../mujoco_tests/mujoco_menagerie/trossen_wx250s/scene.xml", "trossen_wx250s"],
    ["../../mujoco_tests/mujoco_menagerie/trs_so_arm100/scene.xml", "trs_so_arm100"],
    ["../../mujoco_tests/mujoco_menagerie/ufactory_xarm7/scene.xml", "ufactory_xarm7"],
    ["../../mujoco_tests/mujoco_menagerie/universal_robots_ur10e/scene.xml", "universal_robots_ur10e"],
    ["../../mujoco_tests/mujoco_menagerie/universal_robots_ur5e/scene.xml", "universal_robots_ur5e"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_a1/scene.xml", "unitree_a1"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_go1/scene.xml", "unitree_go1"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_go2/scene.xml", "unitree_go2"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_h1/scene.xml", "unitree_h1"],
    ["../../mujoco_tests/mujoco_menagerie/unitree_z1/scene.xml", "unitree_z1"],
]

# Results will be a list of tuples: (cfg, status, scene)
results = []

def print_results():
    print("\n--- Test Summary ---")
    # Group results by configuration
    summary = {}
    for cfg, status, scene in results:
        key = f"{cfg['internal']}   {cfg['format']}   {cfg['type']}"
        if key not in summary:
            summary[key] = {"Passed": [], "FAILED": []}
        if status.upper() == "PASSED":
            summary[key]["Passed"].append(scene)
        else:
            summary[key]["FAILED"].append(scene)
    
    summary_str = ""
    for config, res in summary.items():
        summary_str += f"Results for config {config}:\n"
        if not res["FAILED"]:
            summary_str += "    ALL SCENES PASS\n"
        else:
            if res["Passed"]:
                summary_str += "    Passing scenes:\n"
                for scene in res["Passed"]:
                    summary_str += f"         {scene}\n"
            if res["FAILED"]:
                summary_str += "    Failing scenes:\n"
                for scene in res["FAILED"]:
                    summary_str += f"         {scene}\n"
        summary_str += "\n"
    
    print(summary_str)
    
    # Save the summary to a log file with the timestamp HHMMSS
    timestamp = time.strftime("%H%M%S")
    filename = f"investigation_{timestamp}.log"
    with open(filename, "w") as f:
        f.write(summary_str)
    print(f"Summary saved to {filename}")

for cfg in configs:
    # Generate the include file
    description = f"Testing {cfg['internal']}   {cfg['format']}   {cfg['type']}..."
    line0 = f"wanted_internal_format = {cfg['internal']};"
    line1 = f"glTexImage2D(GL_TEXTURE_2D, 0, wanted_internal_format, con->shadowSize, con->shadowSize, 0, {cfg['format']}, {cfg['type']}, NULL);"
    line2 = f'printf("{description}\\n");'
    with open("../src/render/test_config.inc", "w") as f:
        f.write(line0 + "\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    
    if debug_glTexImage2D_commands:
        print(line1)
        continue
        
    # Compile the program
    try:
        if platform.system() == "Windows":
            # Only works from Native Tools Command Prompt
            subprocess.run("msbuild Mujoco.sln /t:simulate /p:Configuration=Release", shell=True, check=True)
        else:
            subprocess.run("make simulate", shell=True, check=True)
    except:
        print("Failed to build simulate target")
        quit()
        
    for path, name in scenes:
        env = os.environ.copy()
        env["MUJOCO_GL_DEBUG"] = "1"  # Set MUJOCO_GL_DEBUG=1 for all platforms
        try:
            if platform.system() == "Windows":
                subprocess.run(f'bin\Release\simulate.exe "{path}"', shell=True, check=True, env=env)
            else:
                subprocess.run(f'./bin/simulate "{path}"', shell=True, check=True, env=env)
        except:
            print("Failed to run simulate target")
            quit()
        
        choice = get_choice()
        if choice in ('p', 'P'):
            results.append((cfg, "Passed", name))
            if choice == 'P':
                break
        elif choice in ('f', 'F'):
            results.append((cfg, "FAILED", name))
            if choice == 'F':
                break
        elif choice == 'q':
            print_results();
            print("Aborted.\n")
            quit()

print_results()
