def nourishment(name: str, states_definitions: str):
    return f"""\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        Pickup:
            TNT1 A 0 ACS_NamedExecuteAlways("func_{name}")
            Stop
        {states_definitions}
        }}
}}"""


def poison(name: str, states_definitions: str):
    return nourishment(name, states_definitions)


def obstacle(name: str, states_definitions: str, radius: int = 24):
    return f"""\
ACTOR {name} : TorchTree {{
    Radius {radius}

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}"""


def predator(name: str, states_definitions: str, speed: int):
    return f"""\
ACTOR {name} : Actor {{
    Speed {speed}
    States {{
        {states_definitions}\
    }}
}}"""

def distractor(name: str, states_definitions: str):
    return f"""\
ACTOR {name} : CustomInventory {{
    +INVENTORY.ALWAYSPICKUP

    // Different states used for different appearances, randomly chosen at spawn
    States {{
        {states_definitions}\
    }}
}}
"""


def states_template(index: int, texture_code: str):
    return f"Texture{index}: {texture_code} A -1\n\t"


def predator_states_template(index: int, texture_code: str, actor_name: str):
    chase_label = f"Chase{index}"
    melee_label = f"Melee{index}"
    return f"""Texture{index}:
        {texture_code} A 10 A_Look
        Goto {chase_label}
    {chase_label}:
        {texture_code} A 4 A_Chase("{melee_label}", "")
        Loop
    {melee_label}:
        {texture_code} A 0 ACS_NamedExecuteAlways("func_{actor_name}")
        Goto {chase_label}\n\t"""


def include(actor_name: str):
    return f'#include "actors/{actor_name}.dec"\n'
