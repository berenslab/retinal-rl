"""
This module contains a dummy for a vizdoom config.
"""


def config(scenario_name: str):
    """Returns a config for a vizdoom game,
    referencing {scenario_name}.zip as the scenario."""

    return f"""\
doom_scenario_path = {scenario_name}.zip

living_reward = 0.0
death_penalty = 0.0

# Rendering options
screen_resolution = RES_320X240
screen_format = CRCGCB
render_hud = false
render_crosshair = false
render_weapon = false
render_decals = false
render_particles = false
window_visible = false
sound_enabled = false

# make episodes finish after x frames
episode_timeout = 20000

# Available buttons
available_buttons =
	{{
		TURN_LEFT
		TURN_RIGHT
		MOVE_FORWARD
        MOVE_BACKWARD
	}}

# Game variables that will be in the state
available_game_variables = {{ HEALTH USER17 USER18 USER19}}
# USER17 = HEALTH_GATHERED, USER18 = NUM_NOURISHMENTS, USER19 = NUM_POISONS

mode = PLAYER
"""
