import subprocess
import time

debug_glTexImage2D_commands = False

# TODO Collect a note when the run starts
# TODO Remove requirement to press Return

configs = [
    # Use explicitly sized internal format parameters 
    {"internal": "GL_DEPTH_COMPONENT16", "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_SHORT"},
    {"internal": "GL_DEPTH_COMPONENT24", "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_INT"},
    {"internal": "GL_DEPTH_COMPONENT32", "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_INT"},
    {"internal": "GL_DEPTH_COMPONENT32F", "format": "GL_DEPTH_COMPONENT", "type": "GL_FLOAT"},
    {"internal": "GL_DEPTH24_STENCIL8", "format": "GL_DEPTH_STENCIL", "type": "GL_UNSIGNED_INT_24_8"},
    {"internal": "GL_DEPTH32F_STENCIL8", "format": "GL_DEPTH_STENCIL", "type": "GL_FLOAT_32_UNSIGNED_INT_24_8_REV"},

    # Use unsized/"base" internal formats
    {"internal": "GL_DEPTH_COMPONENT", "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_SHORT"},
    {"internal": "GL_DEPTH_COMPONENT", "format": "GL_DEPTH_COMPONENT", "type": "GL_UNSIGNED_INT"},
    {"internal": "GL_DEPTH_COMPONENT", "format": "GL_DEPTH_COMPONENT", "type": "GL_FLOAT"},
]

scenes = [
    ["../../mujoco_tests/mink-main/examples/kuka_iiwa_14/scene.xml", "kuka_iiwa_14"],
    ["../../mujoco_tests/mink-main/examples/stanford_tidybot/scene.xml", "stanford_tidybot"],
    ["../../mujoco_tests/mink-main/examples/unitree_h1/scene.xml", "unitree_h1"],
    ["../../mujoco_tests/mink-main/examples/apptronik_apollo/scene.xml", "apptronik_apollo"],  # Note: Robot falls through the floor
    ["../../mujoco_tests/mink-main/examples/unitree_go1/scene.xml", "unitree_go1"],
    ["../../mujoco_tests/mink-main/examples/universal_robots_ur5e/scene.xml", "universal_robots_ur5e"],
    ["../../mujoco_tests/mink-main/examples/boston_dynamics_spot/scene.xml", "boston_dynamics_spot"],
    ["../../mujoco_tests/mink-main/examples/aloha/scene.xml", "aloha"],
    ["../../mujoco_tests/mink-main/examples/unitree_g1/scene.xml", "unitree_g1"],
    ["../../mujoco_tests/mink-main/examples/ufactory_xarm7/scene.xml", "ufactory_xarm7"],
    # This scene doesn't load: "Error: mesh volume is too small: base_link_2 . Try setting inertia to shell"
    # ["../../mujoco_tests/mink-main/examples/hello_robot_stretch_3/scene.xml", "hello_robot_stretch_3"],
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
    line0 = f"GLint wanted_internal_format = {cfg['internal']};"
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
        subprocess.run("make simulate", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command execution for {cfg['internal']} failed: {e}")
        
    for path, name in scenes:
        try:
            subprocess.run(f'./bin/simulate "{path}"', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command execution for {cfg['internal']} failed: {e}")
        
        user_input = input(
            "Did it work?\n"
            "(p)ass. Try next scene\n"
            "(f)ail. Try next scene\n"
            "(P)ASS. Skip all following scenes\n"
            "(F)AIL. Skip all following scenes\n"
            "(q)uit.\n"
        ).strip()
        if user_input:
            match user_input[0]:
                case 'p': 
                    results.append((cfg, "Passed", name))
                case 'P': 
                    results.append((cfg, "Passed", name))
                    break
                case 'f':
                    results.append((cfg, "FAILED", name))
                case 'F':
                    results.append((cfg, "FAILED", name))
                    break
                case 'q':
                    print_results()
                    print("Aborted.\n")
                    quit()

print_results()
