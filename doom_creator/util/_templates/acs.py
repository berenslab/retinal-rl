from typing import List


def general(
    metabolic_delay: int,
    metabolic_damage: int,
    object_variables: str,
    array_variables: str,
    actor_functions: str,
    spawn_relative: bool = True,
    spawn_range: float = 1000.0,
):
    return f"""\
// Directives
#import "acs/retinal.acs"
#include "zcommon.acs"

script "Load Config Information" OPEN {{

    // Metabolic variables
    metabolic_delay = {metabolic_delay};
    metabolic_damage = {metabolic_damage};

    // Arena settings
    // Spawn behaviour
    spawn_relative = {str(spawn_relative).lower()};
    spawn_range = {spawn_range};

    // Object Variables
    {object_variables}

    // Loading arrays
    {array_variables}
}}

// Actor Functions / Scripts
{actor_functions}"""


def object_variables(typ: str, unique: int, init: int, delay: int):
    return f"""
    // {typ} variables
    {typ}_unique = {unique};
    {typ}_init = {init};
    {typ}_delay = {delay};
"""


def actor_function(actor_name: str, values: List[int], heal_or_damage: bool):
    actor_name = actor_name.replace("-", "_")
    # Actor name will be used in function name, not possible with -
    num_values = len(values)
    values_string = ",".join([str(v) for v in values])

    return f"""\
int values_{actor_name}[{num_values}] = {{ {values_string} }};
script "func_{actor_name}" (void)
{{
    int i = Random(0,{num_values}-1);
    {heal_or_damage}Thing(values_{actor_name}[i]);
}}
"""


def heal_function(actor_name: str, values: List[int]):
    return actor_function(actor_name, values, "Heal")


def damage_function(actor_name: str, values: List[int]):
    return actor_function(actor_name, values, "Damage")


def actor_arrays(index: int, actor_name: str, num_textures: int):
    return f"""
    actor_names[{index}] = "{actor_name}";
    actor_num_textures[{index}] = {num_textures};
"""
