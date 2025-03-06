def convert_mag_to_current(df):

    

    df = df.copy()

    df['field_kg'] = df['field_T'] / 10

    currents = []

    for _, row in df.iterrows():

        field_target, mag = row['field_kg'], row['mag_number']

        if mag not in [1, 2, 3, 4, 5]:

            currents.append(-1)

            continue

        x, tol, h = 1.0, 1e-9, 1e-6

        for _ in range(1000):

            cn = x / 300.0

            if mag == 1:

                fx = (((0.0110605 + 5.33237 * cn - 0.0142794 * cn**2 + 0.259313 * cn**3) +

                       0.0058174 * cn**4 - 0.831887 * cn**5) * 10 * 5.08 / 36.5723)

            elif mag == 2:

                fx = (((0.0196438 + 5.35443 * cn + 0.0297273 * cn**2 + 0.103505 * cn**3) -

                       0.0449275 * cn**4 - 0.211868 * cn**5) * 10 * 5.08 / 44.76)

            elif mag == 3:

                fx = (((0.000632446 + 5.15178 * cn - 0.00262778 * cn**2) +

                       (-0.107635 * cn**3 + 0.00209902 * cn**4) - 0.640635 * cn**5) *

                      10 * 5.08 / 36.74)

            elif mag == 4:

                fx = (((0.0001732 + 5.2119 * cn - 0.000732518 * cn**2) +

                       (-0.133423 * cn**3 + 0.000618402 * cn**4) - 0.647082 * cn**5) *

                      10 * 5.08 / 36.50)

            else:  # mag == 5

                fx = -0.39026e-04 + 0.027051 * x - 0.17799e-08 * x**2

            f_val = field_target - fx

            xp, xm = x + h, x - h

            cnp, cnm = xp / 300.0, xm / 300.0

            if mag == 1:

                fxp = (((0.0110605 + 5.33237 * cnp - 0.0142794 * cnp**2 + 0.259313 * cnp**3) +

                        0.0058174 * cnp**4 - 0.831887 * cnp**5) * 10 * 5.08 / 36.5723)

                fxm = (((0.0110605 + 5.33237 * cnm - 0.0142794 * cnm**2 + 0.259313 * cnm**3) +

                        0.0058174 * cnm**4 - 0.831887 * cnm**5) * 10 * 5.08 / 36.5723)

            elif mag == 2:

                fxp = (((0.0196438 + 5.35443 * cnp + 0.0297273 * cnp**2 + 0.103505 * cnp**3) -

                        0.0449275 * cnp**4 - 0.211868 * cnp**5) * 10 * 5.08 / 44.76)

                fxm = (((0.0196438 + 5.35443 * cnm + 0.0297273 * cnm**2 + 0.103505 * cnm**3) -

                        0.0449275 * cnm**4 - 0.211868 * cnm**5) * 10 * 5.08 / 44.76)

            elif mag == 3:

                fxp = (((0.000632446 + 5.15178 * cnp - 0.00262778 * cnp**2) +

                        (-0.107635 * cnp**3 + 0.00209902 * cnp**4) - 0.640635 * cnp**5) *

                       10 * 5.08 / 36.74)

                fxm = (((0.000632446 + 5.15178 * cnm - 0.00262778 * cnm**2) +

                        (-0.107635 * cnm**3 + 0.00209902 * cnm**4) - 0.640635 * cnm**5) *

                       10 * 5.08 / 36.74)

            elif mag == 4:

                fxp = (((0.0001732 + 5.2119 * cnp - 0.000732518 * cnp**2) +

                        (-0.133423 * cnp**3 + 0.000618402 * cnp**4) - 0.647082 * cnp**5) *

                       10 * 5.08 / 36.50)

                fxm = (((0.0001732 + 5.2119 * cnm - 0.000732518 * cnm**2) +

                        (-0.133423 * cnm**3 + 0.000618402 * cnm**4) - 0.647082 * cnm**5) *

                       10 * 5.08 / 36.50)

            else:  # mag == 5

                fxp = -0.39026e-04 + 0.027051 * xp - 0.17799e-08 * xp**2

                fxm = -0.39026e-04 + 0.027051 * xm - 0.17799e-08 * xm**2

            d_val = (fxp - fxm) / (2 * h)

            if d_val == 0:

                break

            x_new = x + f_val / d_val

            if abs(x_new - x) < tol:

                x = x_new

                break

            x = x_new

        currents.append(x)

    df['current'] = currents

    return df

