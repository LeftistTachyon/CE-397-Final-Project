from powerworld_aux import load_grid

if __name__ == "__main__":
    grid = load_grid("Hawaii40_20231026.zip", matpower=True)
    dc = grid.dc_inputs()

    battery_bus = 2
    battery_index = dc["bus_index"][battery_bus]

    print(battery_index)
