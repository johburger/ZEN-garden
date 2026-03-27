def parameter_change_log():
    log_dict = {
        "min_full_load_hours_fraction": {
            "default_value": 0,  # only 0, 1, or 'inf' are allowed
            "unit": "min_load",
        },
        "min_energy_production": {
            "default_value": 0,
            "unit": "availability_import_yearly"
        },
        "min_total_protein_production": {
            "default_value": 0,
            "unit": "methane_intensity_carrier_export"
        }
        #    "new_parameter_name": {
        #       "default_value": 0, # only 0, 1, or 'inf' are allowed
        #       "unit": "existing_parameter_name_with_same_unit"
        #   }
    }

    return log_dict
