# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
import datetime as dt

import eyecontact as cs

cs.logs(show_level='debug', show_color=True)
logger = cs.CustomLogger(__name__)  # use custom logger

# Const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = True  # update mapping with keypress data
RES = 100  # resolution of keypress data plots
MIN_DUR = -1  # minimal allowed length of stimulus. -1 for any video length
MAX_DUR = -1  # maximal allowed length of stimulus. -1 for any video length
file_coords = 'coords.p'  # file to save lists with coordinates
file_mapping = 'mapping.p'  # file to save lists with coordinates

if __name__ == '__main__':
    # todo: add descriptions for methods automatically with a sublime plugin
    # create object for working with heroku data
    files_heroku = cs.common.get_configs('files_heroku')
    heroku = cs.analysis.Heroku(res=RES,
                                files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = cs.common.get_configs('file_appen')
    appen = cs.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data()
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
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        stimuli_mapped = heroku.read_mapping()
        # read in mapping of stimuli
        stimuli_mapped = heroku.process_kp(min_dur=MIN_DUR, max_dur=MAX_DUR)
        cs.common.save_to_p(file_mapping,
                            stimuli_mapped,
                            'mapping with keypress data')
    else:
        stimuli_mapped = cs.common.load_from_p(file_mapping,
                                               'mapping of stimuli')
    # Output
    analysis = cs.analysis.Analysis(res=RES)
    logger.info('Creating figures.')
    # all keypresses
    analysis.plot_kp(stimuli_mapped)
    # keypresses of an individual stimulus
    analysis.plot_kp_video(stimuli_mapped, 'video_0')
    # keypresses of all videos individually
    analysis.plot_kp_videos(stimuli_mapped)
    # start of eye contact
    analysis.plot_kp_variable(stimuli_mapped, 'start_ec')
    # start of eye contact, certain values
    analysis.plot_kp_variable(stimuli_mapped, 'start_ec', [16.6, 12.54])
    # end of eye contact
    analysis.plot_kp_variable(stimuli_mapped, 'end_ec')
    # separate plots for multiple variables
    analysis.plot_kp_variables_or(stimuli_mapped, [{'variable': 'yielding', 'value': 1},  # noqa: E501
                                                   {'variable': 'start_ec', 'value': 16.6},  # noqa: E501
                                                   {'variable': 'end_ec', 'value': 27.3}])  # noqa: E501
    # multiple variables as a single filter
    analysis.plot_kp_variables_and(stimuli_mapped, [{'variable': 'yielding', 'value': 1},  # noqa: E501
                                                    {'variable': 'start_ec', 'value': 12.54}])  # noqa: E501
    # create correlation matrix
    analysis.corr_matrix(stimuli_mapped, save_file=True)
    # stimulus durations for all participants
    analysis.hist_stim_duration(heroku_data, nbins=100, save_file=True)
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
                    }
                   ]
    analysis.hist_stim_duration_time(all_data,
                                     time_ranges=time_ranges,
                                     nbins=100,
                                     save_file=True)
    # browser window dimensions
    # analysis.hist_browser_dimensions(heroku_data, nbins=100, save_file=True)
    analysis.scatter_questions(heroku_data,
                               x='window_width',
                               y='window_height',
                               color='browser_name',
                               save_file=True)
    analysis.heatmap_questions(heroku_data,
                               x='window_width',
                               y='window_height',
                               save_file=True)
    # time of participation
    analysis.hist_time_participation(appen_data, save_file=True)
    # questions
    analysis.scatter_questions(appen_data,
                               x='ec_driver',
                               y='ec_pedestrian',
                               color='year_license',  # noqa: E501
                               save_file=True)
    # time of participation
    analysis.hist_time_participation(appen_data, save_file=True)
    # check if any figures are to be rendered
    figures = [manager.canvas.figure
               for manager in
               matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    # show figures, if any
    if figures:
        plt.show()
