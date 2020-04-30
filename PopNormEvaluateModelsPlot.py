# Load Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from myutils import check_directory,user_yes_no_query
import sys

class PopNormEvaluateModelsPlots:
    g_extend = 30
    #-------------------- Directories to Save File -------------
    dir_plot = "PN-Plots/"
    dir_plot_extnd = "PN-Plots/ExtendedPred/"
    dir_plot_model = "PN-Plots/ModelPlots/"
    dir_pred = "PN-Predictions/"
    dir_pred_extnd = "PN-Predictions/ExtendedPred/"
    dir_dataset = "Data/Dataset/"

    def __init__(self, extend=30):
        self.g_extend = extend
        self.models=[]
        self.countries = ["AU", "USA", "UK", "Spain", "S.Korea", "Italy", "Germany", "France", "Global"]
        self.country_pop = [25500, 331003, 67886, 46755, 51269,   60462,   83784,   65274,  7794799] #in Thousands
        self.country_pop_norm = [60462.0/x for x in self.country_pop] #normalsied by italy
    # end function

    def assign_variables(self, string):
        '''Define Function: to Replace column with Variable Name'''
        v = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        for idx in range(0, 14):
            var = v[idx] + "2"
            nval = "D" + str(idx + 1)
            string = string.replace(var, nval)
        # end for
        return string
    # end function


    def apply_model(self, model, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14):
        '''Define Function: to apply the model on given data'''
        val = eval(model)
        return val
    # end function

    def create_data_column(self, this_date, model_name, confirmed, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11,
                           D12, D13, D14, Pred):
        '''Define Function: to create a new dataframe'''

        # Define a dictionary containing Students data
        data = {'IDX': this_date, 'Confirmed': confirmed,
                'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4, 'D5': D5, 'D6': D6, 'D7': D7,
                'D8': D8, 'D9': D9, 'D10': D10, 'D11': D11, 'D12': D12, 'D13': D13, 'D14': D14,
                model_name: Pred}

        # Convert the dictionary into DataFrame
        df = pd.DataFrame(data)
        df.set_index("IDX", inplace=True)
        df.rename_axis(None, inplace=True)
        return df
    # end function

    # --------------- Create Directory for Plots & Results
    def check_directories(self, dbg=False):
        #-- Check Directories for Plots
        check_directory(self.dir_plot, dbg=dbg)
        check_directory(self.dir_plot_model, dbg=dbg)
        check_directory(self.dir_plot_extnd, dbg=dbg)

        # -- Check Directories for Predictions
        check_directory(self.dir_pred, dbg=dbg)
        check_directory(self.dir_pred_extnd, dbg=dbg)

        # -- Check Directories for Dataset
        check_directory("Data/", dbg=dbg)
        check_directory(self.dir_dataset, dbg=dbg)

        print("Directories Check Pass!")
    # end function

    def read_str_models(self, fmodels="cfr-10-Models.txt"):
        '''Read 10 models from the text file. The model text for Excel is required.'''
        fh = open(fmodels)
        self.models = [line.rstrip() for line in fh.readlines()]
        fh.close()
    #end function

    def plot_model_pred(self, data, model_col, country, is_extend=False):
        fig = plt.figure(figsize=(12, 5))
        plt.style.use('classic')

        # -- line 1 points
        x1 = data.index.values
        y1 = data["Confirmed"]
        # plotting the line 1 points
        plt.plot(x1, y1, 'r.-', linewidth=1, label="Confirmed")

        # -- line 2 points
        x2 = x1
        y2 = data[model_col]

        # plotting the line 2 points
        plt.plot(x2, y2, 'bo', linewidth=1, markersize=4, label=model_col)

        # -- Plot axis Labels
        plt.xlabel('Date')
        plt.ylabel('Confirmed Cases')
        plt.title("Prediction of Cumulative Confirmed Cased of " + country + " by " + model_col)
        plt.legend(loc='upper left')

        # -- Save the figure.
        filename = ""
        if is_extend: filename = self.dir_plot_extnd + "pn-plt-Extended-Pred-" + country + "-" + model_col + ".pdf"
        else: filename = self.dir_plot_model + "pn-plt-Pred-" + country + "-" + model_col + ".pdf"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        # sys.exit(0)
    #end plot functions

    def plot_overlay_lines_median(self, df_med, df_conf, country):
        # fig, ax = plt.subplots()
        # txt_title = "Prediction of the Median of 10 Models for Confirmed Cases of " + country
        #
        # plt_data = pd.DataFrame()
        # plt_data["Models-Median"] = df_med.median(axis=0)
        # plt_data["Confirmed"] = df_conf["Confirmed"]
        # print(plt_data.head())
        # styles = ['b.', 'r.-']  # , 'y^-']
        # ax = plt_data.plot(style=styles, logy=True, figsize=(12, 5))
        # cmb_plt_file = self.dir_plot + "pn-plt-Extended-Pred-" + country + "-MedLine-Log.pdf"
        # plt.savefig(cmb_plt_file, bbox_inches='tight')
        # plt.close(fig)

        fig = plt.figure(figsize=(12, 5))
        plt.style.use('classic')
        ax = fig.add_subplot(111)

        plt_data = pd.DataFrame()
        plt_data["Models-Median"] = df_med.median(axis=0)
        plt_data["Confirmed"] = df_conf["Confirmed"]

        # -- line 1 points
        x1 = plt_data.index.values
        y1 = plt_data["Confirmed"]
        # plotting the line 1 points
        ax.plot(x1, y1, 'r.-', linewidth=1, label="Confirmed")

        # -- line 2 points
        x2 = x1
        y2 = plt_data["Models-Median"]
        # plotting the line 2 points
        ax.plot(x2, y2, 'bo', markersize=4, label="Models-Median")

        # -- Plot axis Labels
        plt.yscale('log')
        plt.xlabel('Date')
        plt.ylabel('Confirmed Cases (in log scale)')
        plt.title("Median Prediction of 10 Models for Cumulative Confirmed Cases of " + country )
        plt.legend(loc='best') #upper left

        # Customize the tick marks and turn the grid on
        ax.minorticks_on()
        # ax.tick_params(which='major', length=14, width=2, direction='inout')
        ax.tick_params(which='minor', length=2, width=1, direction='in')
        # ax.grid(which='both')

        cmb_plt_file = self.dir_plot + "pn-plt-Extended-Pred-" + country + "-MedLine-Log.pdf"
        plt.savefig(cmb_plt_file, bbox_inches='tight')
        plt.close(fig)

        # sys.exit(0)
    #end function

    def plot_overlay_lines_box(self, df_box, df_line, country, is_log=False,  is_extend=False):
        fig, ax = plt.subplots(figsize=(14, 5))

        # make log scale
        if is_log :
            ax.set_ylabel('Confirmed Cases (in log scale)')
            ax.set_yscale("log", nonposy='clip')
            txt_title = "Boxplot to show the Prediction of 10 Models for Confirmed Cases (in log scale) of " + country
        else:
            txt_title = "Boxplot to show the Prediction of 10 Models for Confirmed Cases of " + country

        # create shared y axes
        ax2 = ax.twiny()
        boxplot = df_box.plot(kind='box', grid=False, fontsize=8, title=txt_title, ax=ax)
        df_line.plot(kind='line', style='r.-', linewidth=1, markersize=5, ax=ax2)
        plt.setp(boxplot.get_xticklabels(), rotation=90)

        x_labels = df_box.T.index.strftime("%Y-%m-%d")
        ax.set_xticklabels(x_labels)

        # remove upper axis ticklabels
        ax2.set_xticklabels([])
        # set the limits of the upper axis to match the lower axis ones
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.legend(loc='upper left')  # , frameon=False)

        cmb_plt_file= self.dir_plot
        if is_extend: cmb_plt_file = cmb_plt_file + "pn-plt-Extended-Pred-" + country + "-BoxLine-Log.pdf"
        else: cmb_plt_file = cmb_plt_file + "pn-plt-Pred-" + country + "-BoxLine-Log.pdf"

        fig.savefig(cmb_plt_file, bbox_inches='tight')
        plt.close(fig)
        # sys.exit(0)
    #end function

    def run(self, dbg=False):
        # Check Directories & load models
        self.check_directories(dbg=dbg)
        self.read_str_models()

        # --- Process Models for each Country
        for country in self.countries:

            country_pred_data = pd.DataFrame()
            # --- read data for the country
            file = self.dir_dataset + "DS-non0-" + country + ".csv"
            data_orig = pd.read_csv(file, index_col=0, parse_dates=True)

            # --- Apply Normalisation of the data according to pop-norm factor by Italy
            pos = self.countries.index(country)
            pn_factor = self.country_pop_norm[pos]
            data_norm = data_orig/pn_factor

            # --- apply each of the models to the country data
            xls_path = self.dir_pred + "Extended-Pred-" + country + ".xlsx"
            writer = pd.ExcelWriter(xls_path, engine='xlsxwriter')

            for model_idx in range(1, len(self.models) + 1):
                data = data_norm.copy()
                dates = data.index.values
                last_date = pd.to_datetime(dates[-1])

                # -- prepare model with variable
                test_model = self.assign_variables(self.models[model_idx - 1])
                model_col = 'Model-' + str(model_idx)

                # -- apply model to the dataset
                data[model_col] = data.apply(lambda row: self.apply_model(test_model, row['D1'], row['D2'], row['D3'],
                                                                       row['D4'], row['D5'], row['D6'], row['D7'],
                                                                       row['D8'], row['D9'], row['D10'], row['D11'],
                                                                       row['D12'], row['D13'], row['D14']), axis=1)

                # --- Plot the Model's Prediction
                self.plot_model_pred(data, model_col, country)

                #------ Prepare the Dataset for the Extended period's Predictions
                last_date += datetime.timedelta(days=1)
                row_l = data[-1:]   #last row of the dataset
                pred = self.apply_model(test_model, row_l['Confirmed'], row_l['D1'], row_l['D2'], row_l['D3'],
                                        row_l['D4'], row_l['D5'], row_l['D6'], row_l['D7'], row_l['D8'], row_l['D9'],
                                        row_l['D10'], row_l['D11'], row_l['D12'], row_l['D13'])

                new_row = self.create_data_column(last_date, model_col, None, row_l['Confirmed'], row_l['D1'],
                                                  row_l['D2'], row_l['D3'], row_l['D4'], row_l['D5'], row_l['D6'],
                                                  row_l['D7'], row_l['D8'], row_l['D9'], row_l['D10'], row_l['D11'],
                                                  row_l['D12'], row_l['D13'], pred)
                # Concat new row
                frames = [data, new_row]
                data = pd.concat(frames)


                for day in range(1, self.g_extend):
                    last_date += datetime.timedelta(days=1)
                    row_l = data[-1:]
                    pred = self.apply_model(test_model, row_l[model_col], row_l['D1'], row_l['D2'], row_l['D3'],
                                            row_l['D4'], row_l['D5'], row_l['D6'], row_l['D7'], row_l['D8'],
                                            row_l['D9'], row_l['D10'], row_l['D11'], row_l['D12'], row_l['D13'])

                    new_row = self.create_data_column(last_date, model_col, None, row_l[model_col], row_l['D1'],
                                                      row_l['D2'], row_l['D3'], row_l['D4'], row_l['D5'], row_l['D6'],
                                                      row_l['D7'], row_l['D8'], row_l['D9'], row_l['D10'], row_l['D11'],
                                                      row_l['D12'], row_l['D13'], pred)
                    frames = [data, new_row]
                    data = pd.concat(frames)
                # end for day

                # Save Extended Predictions into CSV
                if dbg: print("Data Saved After " + str(self.g_extend) + " days of Predictions:: " + model_col)
                data.to_csv(self.dir_pred_extnd + "DS-Extended-Pred-" + country + "-" + model_col + ".csv", index=True)
                country_pred_data[model_col] = data[model_col]

                # Exporting to Excel
                data.to_excel(writer, sheet_name=model_col, index=True)

                # --- Create Plots for each model
                self.plot_model_pred(data, model_col, country, is_extend=True)

                #-- append country data
                country_pred_data["IDX"] = data.index.values
                country_pred_data.set_index("IDX", inplace=True)
                country_pred_data.rename_axis(None, inplace=True)

            # end for model_idx

            # --- Save Extended Predictions Per Country into a CVS file
            country_pred_data['Confirmed'] = data_orig['Confirmed']
            country_pred_data.to_csv(self.dir_pred + "Extend-Preds-Models-" + country + ".csv", index=True)

            # Exporting to Excel
            country_pred_data["Average"]=country_pred_data.mean(axis=1)
            country_pred_data["Median"] = country_pred_data.median(axis=1)

            country_pred_data.to_excel(writer, sheet_name="AllModels", index=True)
            writer.save()
            writer.close()

            # ---------------------------- Plots --------------------------
            box_data = country_pred_data.copy()
            del box_data["Confirmed"]
            df = box_data.T
            line_data = pd.DataFrame()
            line_data["Confirmed"] = country_pred_data["Confirmed"]
            df_confirmed = line_data

            # --- Overlay Box and Line
            self.plot_overlay_lines_box(df, df_confirmed, country, is_log=False, is_extend=True)

            # --- Overlay Box and Line (log Scale)
            self.plot_overlay_lines_box(df, df_confirmed, country, is_log=True, is_extend=True)

            # --- Overlay Lines (log Scale) Median
            self.plot_overlay_lines_median(df, df_confirmed, country)

            print("Done Plotting for: " + country + "\n")
            sys.exit(0)
        # end for country
    #end function run
#end class


if __name__ == '__main__':
    if user_yes_no_query("Warning!!!\nThis Execution will replace existing figures. Do you want to run now?")==True:
        PopNormEvaluateModelsPlots().run(True)
    else:
        print("Thank you for deciding to keep existing figures intact.")
