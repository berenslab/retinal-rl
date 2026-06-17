from typing import List


def general(
    metabolic_delay: int,
    metabolic_damage: int,
    object_variables: str,
    array_variables: str,
    actor_functions: str,
    spawn_relative: bool = True,
    spawn_range: int = 1000,
    cell_width: int = 150,
):
    grid_size = spawn_range * 2 // cell_width
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
    grid_size = {grid_size};
    {""
    "".join([f"free_positions[{i}]=1;" for i in range(grid_size**2)])};
    objects_left_to_spawn = {grid_size**2};

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


def actor_function(actor_name: str, values: List[int], heal_or_damage: str):
    actor_name = actor_name.replace("-", "_")
    # Actor name will be used in function name, not possible with -
    num_values = len(values)
    values_string = ",".join([str(v) for v in values])


    plus_minus = "+" if heal_or_damage == "Heal" else "-"
    update_health_gathered = f"health_gathered = health_gathered {plus_minus} values_{actor_name}[i];"

    return f"""\
int values_{actor_name}[{num_values}] = {{ {values_string} }};
script "func_{actor_name}" (void)
{{
    int i = Random(0,{num_values}-1);
    {heal_or_damage}Thing(values_{actor_name}[i]);

    // Update health gathered for stats
    {update_health_gathered}

    // Free space for spawning
    int x = GetActorX(0) >> 16;
    int y = GetActorY(0) >> 16;

    int cell_width = spawn_range/grid_size * 2;
    int grid_x = (x + spawn_range + cell_width/2)/cell_width;
    int grid_y = (y + spawn_range + cell_width/2)/cell_width;
    int grid_index = grid_x*grid_size + grid_y;
    free_positions[grid_index] = 1;
    objects_left_to_spawn++;

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
