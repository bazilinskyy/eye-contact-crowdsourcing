# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import datetime as dt

import eyecontact as cs

cs.logs(show_level='info', show_color=True)
logger = cs.CustomLogger(__name__)  # use custom logger

# Const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = True  # filter Appen and heroku data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = True  # update mapping with keypress data
SHOW_OUTPUT = True  # shoud figures
file_mapping = 'mapping.p'  # file to save lists with coordinates

if __name__ == '__main__':
    # check if config file is updated
    if not cs.common.check_config():
        sys.exit()
    # create object for working with heroku data
    files_heroku = cs.common.get_configs('files_heroku')
    heroku = cs.analysis.Heroku(files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data(filter_data=FILTER_DATA)
    # create object for working with appen data
    file_appen = cs.common.get_configs('file_appen')
    appen = cs.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data(filter_data=FILTER_DATA)
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = cs.analysis.QA(file_cheaters=cs.common.get_configs('file_cheaters'),  # noqa: E501
                            job_id=cs.common.get_configs('appen_job'))
        qa.flag_users()
        qa.reject_users()
    # merge heroku and appen dataframes into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    logger.info('Data from {} participants included in analysis.',
                all_data.shape[0])
    # update original data files
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # generate country-specific data
    countries_data = appen.process_countries()
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        mapping = heroku.read_mapping()
        # process keypresses and update mapping
        mapping = heroku.process_kp()
        # process post-trial questions and update mapping
        questions = [{'question': 'eye_contact', 'type': 'num'},
                     {'question': 'intuitive', 'type': 'num'}]
        mapping = heroku.process_stimulus_questions(questions)
        # export to pickle
        cs.common.save_to_p(file_mapping,
                            mapping,
                            'mapping with keypress data')
    else:
        mapping = cs.common.load_from_p(file_mapping,
                                        'mapping of stimuli')
    if SHOW_OUTPUT:
        # Output
        analysis = cs.analysis.Analysis()
        logger.info('Creating figures.')
        # all keypresses
        analysis.plot_kp(mapping)
        # all keypresses with coonfidence interval, not finished
        analysis.plot_kp_conf_int(mapping)
        # keypresses of an individual stimulus
        analysis.plot_kp_video(mapping, 'video_0')
        # keypresses of all videos individually
        analysis.plot_kp_videos(mapping)
        # start of eye contact
        analysis.plot_kp_variable(mapping, 'start_ec')
        # start of eye contact, certain values
        analysis.plot_kp_variable(mapping, 'start_ec', [16.6, 12.54])
        # end of eye contact
        analysis.plot_kp_variable(mapping, 'end_ec')
        # separate plots for multiple variables
        analysis.plot_kp_variables_or(mapping, [{'variable': 'yielding', 'value': 1},  # noqa: E501
                                                {'variable': 'start_ec', 'value': 16.6},  # noqa: E501
                                                {'variable': 'end_ec', 'value': 27.3}])  # noqa: E501
        # multiple variables as a single filter
        analysis.plot_kp_variables_and(mapping, [{'variable': 'yielding', 'value': 1},  # noqa: E501
                                                 {'variable': 'start_ec', 'value': 12.54}])  # noqa: E501
        # columns to drop in correlation matrix and scatter matrix
        columns_drop = ['no', 'scenario', 'speed', 'video_length', 'kp',
                        'min_dur', 'max_dur']
        # set nan to -1
        df = mapping
        df = df.fillna(-1)
        # create correlation matrix
        analysis.corr_matrix(df,
                             columns_drop=columns_drop,
                             save_file=True)
        # create correlation matrix
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                color='dur_ec',
                                symbol='dur_ec',
                                diagonal_visible=False,
                                save_file=True)
        # stimulus duration
        analysis.hist(heroku_data,
                      x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],  # noqa: E501
                      nbins=100,
                      pretty_text=True,
                      save_file=True)
        # stimulus durations for 2 time periods
        time_ranges = [  # 1st pilot
                       {'start': dt.datetime(2021, 3, 16, 00, 00, 00, 000,
                                             tzinfo=dt.timezone.utc),
                        'end': dt.datetime(2021, 3, 20, 00, 00, 00, 000,
                                           tzinfo=dt.timezone.utc)
                        },
                       # 2nd pilot
                       {'start': dt.datetime(2021, 3, 29, 00, 00, 00, 000,
                                             tzinfo=dt.timezone.utc),
                        'end': dt.datetime(2021, 4, 4, 00, 00, 00, 000,
                                           tzinfo=dt.timezone.utc)
                        },
                       # 3rd pilot
                       {'start': dt.datetime(2021, 4, 8, 00, 00, 00, 000,
                                             tzinfo=dt.timezone.utc),
                        'end': dt.datetime(2021, 4, 10, 00, 00, 00, 000,
                                           tzinfo=dt.timezone.utc)
                        }
                       ]
        analysis.hist_stim_duration_time(all_data,
                                         time_ranges=time_ranges,
                                         nbins=100,
                                         save_file=True)
        # browser window dimensions
        analysis.scatter(heroku_data,
                         x='window_width',
                         y='window_height',
                         color='browser_name',
                         pretty_text=True,
                         save_file=True)
        analysis.heatmap(heroku_data,
                         x='window_width',
                         y='window_height',
                         pretty_text=True,
                         save_file=True)
        # time of participation
        df = appen_data
        df['country'] = df['country'].fillna('NaN')
        analysis.hist(df,
                      x=['time'],
                      color='country',
                      save_file=True)
        # eye contact of driver and pedestrian
        analysis.scatter(appen_data,
                         x='ec_driver',
                         y='ec_pedestrian',
                         color='year_license',
                         pretty_text=True,
                         save_file=True)
        # histogram for driving frequency
        analysis.hist(appen_data,
                      x=['driving_freq'],
                      pretty_text=True,
                      save_file=True)
        # grouped barchart of DBQ data
        analysis.hist(appen_data,
                      x=['dbq1_anger',
                         'dbq2_speed_motorway',
                         'dbq3_speed_residential',
                         'dbq4_headway',
                         'dbq5_traffic_lights',
                         'dbq6_horn',
                         'dbq7_mobile'],
                      marginal='violin',
                      pretty_text=True,
                      save_file=True)
        # bar chart of post-trial eye contact / intuitiveness
        analysis.bar(mapping,
                     y=['eye_contact', 'intuitive'],
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Score',
                     show_text_labels=True,
                     save_file=True)
        # scatter plot of post-trial eye contact / intuitiveness
        df = mapping
        # hardcode +1 for output
        df['intuitive'] = df['intuitive'] + 1
        df['no'] = df['no'] + 1
        print(df[['eye_contact', 'intuitive']])
        analysis.scatter(df,
                         x='eye_contact',
                         y='intuitive',
                         color='dur_ec',
                         # size='yielding',
                         # text='no',
                         trendline='ols',
                         hover_data=['no', 'eye_contact', 'intuitive',
                                     'yielding', 'start_ec', 'end_ec',
                                     'dur_ec'],
                         marker_size=10,
                         pretty_text=True,
                         xaxis_title='Did the driver make eye contact with '
                                     + 'you? (0-1)',
                         yaxis_title='The driver\'s eye contact was intuitive '
                                     + '(1-5)',
                         # xaxis_range=[0.1, 1],
                         # yaxis_range=[2.5, 4],
                         # marginal_x='histogram',
                         # marginal_y='histogram',
                         save_file=True)
        # bar chart of post-trial eye contact
        analysis.bar(mapping,
                     y=['eye_contact'],
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Score',
                     show_text_labels=True,
                     save_file=True)
        # map of participants
        analysis.heatmap_participants(countries_data, save_file=True)
        # map of mean age per country
        analysis.map(countries_data, color='age', save_file=True)
        # map of gender per country
        analysis.map(countries_data, color='gender', save_file=True)
        # map of year of obtaining license per country
        analysis.map(countries_data, color='year_license', save_file=True)
        # map of year of automated driving per country
        analysis.map(countries_data, color='year_ad', save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
