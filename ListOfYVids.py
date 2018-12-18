'''   Produce record of videos where window indicates car/truck could be pulling into drive
      Record if csv contains car/truck instance where x, y, windowsize =
            range of acceptable x, range of acceptable y, range of acceptable window size       '''


# iterate over this function to keep/select CSVs containing
def select_csv(record_name, data, min_rng, max_rng, dec1, dec2):
    car_truck = data.loc[data['label'] != '-']
    # ensure car/truck detected in at least one frame in csv satisfies all conditions
    car_truck = car_truck[(car_truck.topleft_x.astype(int) >= min_rng['topleft_x']) &
                          (car_truck.topleft_x.astype(int) <= max_rng['topleft_x']) &
                          (car_truck.topleft_y.astype(int) >= min_rng['topleft_y']) &
                          (car_truck.topleft_y.astype(int) <= max_rng['topleft_y']) &
                          (car_truck.bttmright_x.astype(int) >= min_rng['bttmright_x']) &
                          (car_truck.bttmright_x.astype(int) <= max_rng['bttmright_x']) &
                          (car_truck.bttmright_y.astype(int) >= min_rng['bttmright_y']) &
                          (car_truck.bttmright_y.astype(int) <= max_rng['bttmright_y'])]
    if car_truck.empty:
        # append number of csv to list of non customers
        dec2.append(int(''.join(filter(str.isdigit, record_name))))
    elif not car_truck.empty:
        # append number of csv to list of customers
        dec1.append(int(''.join(filter(str.isdigit, record_name))))
    return dec1, dec2
