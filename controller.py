import numpy as np
from gymnasium import Space
from gymnasium.core import ActType
import os
from juliacall import Main as jl
from PIL import Image
   
     
class FinalController:
     
    def __init__(self, action_space: Space[ActType]) -> None:
        self.action_space = action_space
        os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
        os.environ['JULIA_NUM_THREADS'] = '1'
        jl.seval("""
		using InteractiveUtils
		println(versioninfo())
        """)
        path = "pongvenv/julia_env/scripts/setup_function_libraries.jl"
        with open(path) as f:
              juliacode = f.read()
        module_path = '"pongvenv/julia_env/src/PongCompetition.jl"'
        utils_path = f'"pongvenv/julia_env/scripts/utils.jl"'
        extra_fns_path = f'"pongvenv/julia_env/scripts/extra_fns.jl"'
        replace_tuples = [('"scripts", "utils.jl"', utils_path),
					('"scripts", "extra_fns.jl"', extra_fns_path),
					('"src/PongCompetition.jl"', module_path)
					]
        juliacode_formatted = juliacode
        for t in replace_tuples:
              juliacode_formatted = juliacode_formatted.replace(t[0], t[1])
        jl.seval(f""" {juliacode_formatted} """)
              
        jl.seval("""
			experimental_horizontal_argmax = ml[2][:experimental_horizontal_argmax].fn
			notmaskfromtoh_image2D = ml[1][:experimental_notmaskfromtoh_image2D].fn
			experimental_opening_image2D_factory = ml[1][:experimental_opening_image2D_factory].fn
			center_of_mass = ml[3][:center_of_mass].fn
			argmin_position = ml[3][:argmin_position].fn
			true_gt = ml[2][:true_gt].fn

			sobely_image2D = ml[1][:sobely_image2D].fn
			experimental_vertical_relative_argmax = ml[2][:experimental_vertical_relative_argmax].fn
			
			experimental_is_gt = ml[2][:experimental_is_gt].fn
			erosion_image2D_factory = ml[1][:erosion_image2D_factory].fn
			experimental_notmaskfromtoh_relative_image2D = ml[1][:experimental_notmaskfromtoh_relative_image2D].fn
			reduce_maximum = ml[2][:reduce_maximum].fn
			
			identity_float = ml[2][:identity_float].fn
		""")
        
        jl.seval("""
			function evolved_pong_policy(f1, f2, f3, f4)
                 
                frame1 = SImageND(f1[CROP[1], CROP[2]], S)
                frame2 = SImageND(f2[CROP[1], CROP[2]], S)
                frame3 = SImageND(f3[CROP[1], CROP[2]], S)
                frame4 = SImageND(f4[CROP[1], CROP[2]], S)
                 				
				# output NO OP
				h_argmax = experimental_horizontal_argmax(frame3) # 33,2
				notmaskfromtoh_img = notmaskfromtoh_image2D(frame3, 0.0, 30.0) # 30,1
				opening_img = experimental_opening_image2D_factory(notmaskfromtoh_img, h_argmax) # 40,1
				center = center_of_mass(frame2) # 48,3
                println("CENTER: $center")
				argmin = argmin_position(opening_img) # 57,3
                println("ARGMIN: $argmin")
				tr_gt = true_gt(center, argmin) # 114,2 
				output_noop = identity_float(tr_gt) # 115 2
                println("output noop : $output_noop")

				# Output RIGHTFIRE (UP)     
				sobely_img = sobely_image2D(frame3, 2.0) # 25,1    
				v_rel_argmax = experimental_vertical_relative_argmax(sobely_img) # 113,2
				output_up = identity_float(v_rel_argmax) # 116 2
                println("output up : $output_up")

				# Output LEFTFIRE (DOWN)
				is_gt1 = experimental_is_gt(50.0, 0.0) # 28,2
				is_gt2 = experimental_is_gt(20.0, is_gt1) # 39,2
				erosion_img = erosion_image2D_factory(frame3, 60.0) # 20,1
				notmaskfromtoh_rel_img = experimental_notmaskfromtoh_relative_image2D(erosion_img, is_gt2, 2.0) # (58,1)
				reduce_max = reduce_maximum(notmaskfromtoh_rel_img) # 112,2)
				output_down = identity_float(reduce_max) # 117 2
                println("output down : $output_down")

				outputs = (output_noop, output_up, output_down)
				return ACTIONS[argmax(outputs)]

			end
			""")
             
    def play(self, frame1, frame2, frame3, frame4):
          jl.frame1 = frame1
          jl.frame2 = frame2
          jl.frame3 = frame3
          jl.frame4 = frame4

          jl.seval("""frame1 = clamp01.(convert.(N0f8, pyconvert(Array, frame1)))""")
          jl.seval("""frame2 = clamp01.(convert.(N0f8, pyconvert(Array, frame2)))""")
          jl.seval("""frame3 = clamp01.(convert.(N0f8, pyconvert(Array, frame3)))""")
          jl.seval("""frame4 = clamp01.(convert.(N0f8, pyconvert(Array, frame4)))""")

          action = jl.seval("""action = evolved_pong_policy(frame1, frame2, frame3, frame4)""")
          print(f"action from play(): {action}")
          return action
             
    def control(self, observation: np.ndarray) -> np.ndarray:
          processed_obs = self.preprocess_observations(observation)

          frame1 = processed_obs[0]
          frame2 = processed_obs[1]
          frame3 = processed_obs[2]
          frame4 = processed_obs[3]

          action = self.play(frame1, frame2, frame3, frame4)
          return action
    

    def preprocess_observations(self, obs_array):
        """
        Preprocess a sequence of 4 observations (images).

        Args:
            obs_array (np.ndarray): Array of shape (4, H, W, C) representing 4 RGB observations.

        Returns:
            np.ndarray: Preprocessed array of shape (4, 84, 84), dtype uint8.
        """
        processed = []
        for obs in obs_array:
            img = Image.fromarray(obs)                          # Convert to PIL Image
            processed.append(np.array(img, dtype=np.float32))   # Convert back to NumPy array
        
        return np.stack(processed)  # Shape: (4, 84, 84)
    
     

class Controller:

    def __init__(self, action_space: Space[ActType]) -> None:
        self.action_space = action_space
        os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'yes'
        os.environ['JULIA_NUM_THREADS'] = '1'
        jl.seval("""
		using InteractiveUtils
		println(versioninfo())
        """)
        path = "pongvenv/julia_env/scripts/setup_function_libraries.jl"
        with open(path) as f:
              juliacode = f.read()
        module_path = '"pongvenv/julia_env/src/PongCompetition.jl"'
        utils_path = f'"pongvenv/julia_env/scripts/utils.jl"'
        extra_fns_path = f'"pongvenv/julia_env/scripts/extra_fns.jl"'
        replace_tuples = [('"scripts", "utils.jl"', utils_path),
					('"scripts", "extra_fns.jl"', extra_fns_path),
					('"src/PongCompetition.jl"', module_path)
					]
        juliacode_formatted = juliacode
        for t in replace_tuples:
              juliacode_formatted = juliacode_formatted.replace(t[0], t[1])
        jl.seval(f""" {juliacode_formatted} """)
              
        jl.seval("""
			experimental_horizontal_argmax = ml[2][:experimental_horizontal_argmax].fn
			notmaskfromtoh_image2D = ml[1][:experimental_notmaskfromtoh_image2D].fn
			experimental_opening_image2D_factory = ml[1][:experimental_opening_image2D_factory].fn
			center_of_mass = ml[3][:center_of_mass].fn
			argmin_position = ml[3][:argmin_position].fn
			true_gt = ml[2][:true_gt].fn

			sobely_image2D = ml[1][:sobely_image2D].fn
			experimental_vertical_relative_argmax = ml[2][:experimental_vertical_relative_argmax].fn
			
			experimental_is_gt = ml[2][:experimental_is_gt].fn
			erosion_image2D_factory = ml[1][:erosion_image2D_factory].fn
			experimental_notmaskfromtoh_relative_image2D = ml[1][:experimental_notmaskfromtoh_relative_image2D].fn
			reduce_maximum = ml[2][:reduce_maximum].fn
			
			identity_float = ml[2][:identity_float].fn
		""")
        
        jl.seval("""
			function evolved_pong_policy(f1, f2, f3, f4)
                 
                frame1 = SImageND(f1[CROP[1], CROP[2]], S)
                frame2 = SImageND(f2[CROP[1], CROP[2]], S)
                frame3 = SImageND(f3[CROP[1], CROP[2]], S)
                frame4 = SImageND(f4[CROP[1], CROP[2]], S)
                 				
				# output NO OP
				h_argmax = experimental_horizontal_argmax(frame3) # 33,2
				notmaskfromtoh_img = notmaskfromtoh_image2D(frame3, 0.0, 30.0) # 30,1
				opening_img = experimental_opening_image2D_factory(notmaskfromtoh_img, h_argmax) # 40,1
				center = center_of_mass(frame2) # 48,3
                println("CENTER: $center")
				argmin = argmin_position(opening_img) # 57,3
                println("ARGMIN: $argmin")
				tr_gt = true_gt(center, argmin) # 114,2 
				output_noop = identity_float(tr_gt) # 115 2
                println("output noop : $output_noop")

				# Output RIGHTFIRE (UP)     
				sobely_img = sobely_image2D(frame3, 2.0) # 25,1    
				v_rel_argmax = experimental_vertical_relative_argmax(sobely_img) # 113,2
				output_up = identity_float(v_rel_argmax) # 116 2
                println("output up : $output_up")

				# Output LEFTFIRE (DOWN)
				is_gt1 = experimental_is_gt(50.0, 0.0) # 28,2
				is_gt2 = experimental_is_gt(20.0, is_gt1) # 39,2
				erosion_img = erosion_image2D_factory(frame3, 60.0) # 20,1
				notmaskfromtoh_rel_img = experimental_notmaskfromtoh_relative_image2D(erosion_img, is_gt2, 2.0) # (58,1)
				reduce_max = reduce_maximum(notmaskfromtoh_rel_img) # 112,2)
				output_down = identity_float(reduce_max) # 117 2
                println("output down : $output_down")

				outputs = (output_noop, output_up, output_down)
				return ACTIONS[argmax(outputs)]

			end
			""")
        
    def play(self, frame1, frame2, frame3, frame4):
          #jlstore = jl.seval("(k, v) -> (@eval $(Symbol(k)) = $v; return)")
          #jlstore("frame1", frame1)
          #jlstore("frame2", frame2)
          #jlstore("frame3", frame3)
          #jlstore("frame4", frame4)

          jl.frame1 = frame1
          jl.frame2 = frame2
          jl.frame3 = frame3
          jl.frame4 = frame4

          jl.seval("""frame1 = clamp01.(convert.(N0f8, pyconvert(Array, frame1)))""")
          jl.seval("""frame2 = clamp01.(convert.(N0f8, pyconvert(Array, frame2)))""")
          jl.seval("""frame3 = clamp01.(convert.(N0f8, pyconvert(Array, frame3)))""")
          jl.seval("""frame4 = clamp01.(convert.(N0f8, pyconvert(Array, frame4)))""")

          action = jl.seval("""action = evolved_pong_policy(frame1, frame2, frame3, frame4)""")
          print(f"action from play(): {action}")
          return action
             
    def control(self, observation: np.ndarray) -> np.ndarray:
          processed_obs = self.preprocess_observations(observation)

          frame1 = processed_obs[0]
          frame2 = processed_obs[1]
          frame3 = processed_obs[2]
          frame4 = processed_obs[3]

          action = self.play(frame1, frame2, frame3, frame4)
          return action
    

    def preprocess_observations(self, obs_array):
        """
        Preprocess a sequence of 4 observations (images).

        Args:
            obs_array (np.ndarray): Array of shape (4, H, W, C) representing 4 RGB observations.

        Returns:
            np.ndarray: Preprocessed array of shape (4, 84, 84), dtype uint8.
        """
        processed = []
        for obs in obs_array:
            img = Image.fromarray(obs)                          # Convert to PIL Image
            img = img.convert("L")                              # Convert to grayscale
            img = img.resize((84, 84))                          # Resize
            img = np.array(img, dtype=np.float32) / 255.0       # Normalize to [0, 1]
            processed.append(np.array(img, dtype=np.float32))   # Convert back to NumPy array
        
        return np.stack(processed)  # Shape: (4, 84, 84)


        
class RandomController(Controller):

    def __init__(self, action_space: Space[ActType]) -> None:
        self.action_space = action_space

    def control(self, observation: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

