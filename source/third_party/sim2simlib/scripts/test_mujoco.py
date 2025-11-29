"""
MuJoCo CPU Visualization Script
A visualization script using MuJoCo CPU version with command-line XML path parameter
"""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os


def main():
    """Main function: Parse command-line arguments and launch MuJoCo visualization"""
    parser = argparse.ArgumentParser(description='MuJoCo CPU Visualization Script')
    parser.add_argument('xml_path', type=str, help='Path to the XML model file')
    parser.add_argument('--timestep', type=float, default=0.002, help='Simulation timestep (default: 0.002)')
    parser.add_argument('--realtime', action='store_true', help='Run in realtime mode')
    
    args = parser.parse_args()
    
    # Check if XML file exists
    if not os.path.exists(args.xml_path):
        print(f"Error: XML file does not exist: {args.xml_path}")
        sys.exit(1)
    
    try:
        # Load model
        print(f"Loading model: {args.xml_path}")
        model = mujoco.MjModel.from_xml_path(args.xml_path)
        data = mujoco.MjData(model)
        
        # Set timestep
        model.opt.timestep = args.timestep
        
        print(f"Model loaded successfully!")
        print(f"Number of DOFs: {model.nv}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of bodies: {model.nbody}")
        print(f"Timestep: {model.opt.timestep}")
        print(f"Data, qpos len: {data.qpos.shape}")
        print(f"Data, qvel len: {data.qvel.shape}")

        # Launch visualization
        print("Launching visualization window...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Visualization window launched! Press ESC or close window to exit")
            print("Control instructions:")
            print("  - Left mouse + drag: Rotate view")
            print("  - Right mouse + drag: Pan view")
            print("  - Mouse wheel: Zoom")
            print("  - Double click: Center view")
            print("  - Space: Pause/resume simulation")
            
            # Main simulation loop
            start_time = time.time()
            step_count = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # Execute simulation step
                mujoco.mj_step(model, data)
                step_count += 1
                
                # Update visualization
                viewer.sync()
                
                # Realtime mode control
                if args.realtime:
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                
                # Print statistics every 1000 steps
                if step_count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    sim_time = data.time
                    print(f"Steps: {step_count}, Sim time: {sim_time:.2f}s, "
                          f"Real time: {elapsed_time:.2f}s, Speed: {sim_time/elapsed_time:.1f}x")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
