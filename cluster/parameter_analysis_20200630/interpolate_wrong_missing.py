# created by Florian Schunck on 09.07.2020
# Project: Tainter
# Short description of the feature:
# 
# 
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
# 
# ------------------------------------------------------------------------------


import numpy as np
import pandas as pd

# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
data = np.loadtxt("output.txt", delimiter=",", skiprows=1)
colnames = np.array(["p_e", "rho", "phi", "te", "st"])
print("import complete")

data_orig = data

# interpolate missing value ----------------------------------------------------

ok = False
while not ok:
    cp = "te"  # check parameter
    r_nan = np.where(np.isnan(data[:, colnames == cp]))[0]
    r_large = np.where(data[:, colnames == cp] > 10e6)[0]
    print("Column:" + cp + "--- nan_rows:", r_nan, "large_rows:", r_large)

    prs = np.concatenate((r_nan, r_large))
    for pr in prs:
        pr_ok = False
        while not pr_ok:
            data_changed = data

            pr_range = np.arange(pr-3, pr+4)
            d = pd.DataFrame(data=data_changed[np.arange(pr-3, pr+4), :],
                             index=pr_range,
                             columns=colnames)
            input_ok = False
            while not input_ok:
                print(d)
                pr_u = int(input("Which value is too large? Maximum should be around 10e5. "
                                 "Enter row number: "))

                print(d.loc[pr_u])
                if bool(input("row ok? Press Enter if false, press any key to continue: ")):
                    input_ok = True

            pars = set(["p_e", "rho", "phi"])
            correct = False
            while not correct:
                try:
                    pr_interp = str(input("enter parameter to interpolate by " +
                                          str(pars) + " : "))
                    pr_filter = pars.difference([pr_interp])
                    correct = True
                    print("thank you!")
                except:
                    print("wrong column name. try again")

            df = pd.DataFrame(data, columns=colnames)
            for fi in pr_filter:
                df = df.loc[df[fi] == df.loc[pr_u, fi]]

            df.sort_values(by=[pr_interp])

            # show selection wrapping the problem line filtered along the proper index
            pr_idx = np.where(df.index == pr_u)[0][0]
            print(df.loc[df.index[(pr_idx-2):(pr_idx+3)]])
            input("continue...")
            te_pr = (df.loc[df.index[pr_idx-1], cp] + df.loc[df.index[pr_idx+1], cp]) / 2
            df.loc[df.index[pr_idx], cp] = te_pr
            print(df.loc[df.index[(pr_idx - 2):(pr_idx + 3)]])
            input("continue...")
            data_changed[pr_u, colnames == cp] = te_pr
            print(pd.DataFrame(data=data_changed[np.arange(pr-3, pr+4), :],
                               index=pr_range,
                               columns=colnames))

            pr_ok = bool(input("all correct? Enter any key if ok, press enter when wrong.: "))
            # TODO: Store output correctly!

        data = data_changed  # store data when correct
    ok = bool(input("Enter 'True' if everything is ok. press enter if not ok"))

n_changed = np.sum(data-data_orig != 0)
print(n_changed, "entries were changed.")

np.savetxt("./output_corrected.txt", data, header=colnames, delimiter=",", newline="\n")

print("saved in output_corrected.txt")