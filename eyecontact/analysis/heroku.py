# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import ast
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import eyecontact as cs

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = cs.CustomLogger(__name__)  # use custom logger


class Heroku:
    # todo: parse browser interactions
    files_data = []  # list of files with heroku data
    heroku_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping'
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'group_choice',
                 'image_ids']
    # prefixes used for files in node.js implementation
    prefixes = {'stimulus': 'video_'}  # noqa: E501
    # stimulus duration
    default_dur = 0

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.files_data = files_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv
        self.num_stimuli = cs.common.get_configs('num_stimuli')

    def set_data(self, heroku_data):
        """
        Setter for the data object.
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self):
        """
        Read data into an attribute.
        """
        # todo: read heroku data
        # load data
        if self.load_p:
            df = cs.common.load_from_p(self.file_p,
                                       'heroku data')
        # process data
        # todo: save browser interaction lists per stimulus
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # last found stimulus
                stim_name = ''
                # trial last found stimulus
                stim_trial = -1
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # extract name of stimulus after last slash
                        # list of stimuli. use 1st
                        if isinstance(data_cell['stimulus'], list):
                            stim_no_path = data_cell['stimulus'][0].rsplit('/', 1)[-1]  # noqa: E501
                        # single stimulus
                        else:
                            stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]  # noqa: E501
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (cs.common.search_dict(self.prefixes, stim_no_path)  # noqa: E501
                           is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['stimulus'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_name = stim_no_path
                                # record trial of stimulus
                                stim_trial = data_cell['trial_index']
                    # keypresses
                    if 'rts' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['rts']
                        logger.debug('Found {} points in keypress data.',
                                     len(responses))
                        # extract pressed keys and rt values
                        key = [point['key'] for point in responses]
                        rt = [point['rt'] for point in responses]
                        # Check if inputted values were recorded previously
                        if stim_name + '-key' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-key'] = key
                        else:
                            # previous values found
                            dict_row[stim_name + '-key'].append(key)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-rt' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-rt'] = rt
                        else:
                            # previous values found
                            dict_row[stim_name + '-rt'].append(rt)
                    # questions
                    if 'responses' in data_cell.keys() and stim_name != '':
                        # record given keypresses
                        responses = data_cell['responses']
                        logger.debug('Found responses to questions {}.',
                                     responses)
                        # extract pressed keys and rt values
                        responses = ast.literal_eval(re.search('({.+})',
                                                     responses).group(0))
                        # unpack questions and answers
                        questions = []
                        answers = []
                        for key, value in responses.items():
                            questions.append(key)
                            answers.append(value)
                        # Check if inputted values were recorded previously
                        if stim_name + '-qs' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qs'] = questions
                        else:
                            # previous values found
                            dict_row[stim_name + '-qs'].append(questions)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-as' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-as'] = answers
                        else:
                            # previous values found
                            dict_row[stim_name + '-as'].append(answers)
                    # question order
                    if 'question_order' in data_cell.keys() \
                       and stim_name != '':
                        # unpack question order
                        qo_str = data_cell['question_order']
                        # remove brackets []
                        qo_str = qo_str[1:]
                        qo_str = qo_str[:-1]
                        # unpack to int
                        question_order = [int(x) for x in qo_str.split(',')]
                        logger.debug('Found question order {}.',
                                     question_order)
                        # Check if inputted values were recorded previously
                        if stim_name + '-qo' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qo'] = question_order
                        else:
                            # previous values found
                            dict_row[stim_name + '-qo'].append(question_order)
                    # injection question
                    if 'injection_q' in data_cell.keys() \
                       and stim_name != '':
                        # record given keypresses
                        injection_q = data_cell['injection_q']
                        logger.debug('Found injection question {}.',
                                     injection_q)
                        # Check if inputted values were recorded previously
                        if stim_name + '-qi' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-qi'] = [injection_q]
                        else:
                            # previous values found
                            dict_row[stim_name + '-qi'].append(injection_q)
                    # browser interaction events
                    if 'interactions' in data_cell.keys() and stim_name != '':
                        interactions = data_cell['interactions']
                        logger.debug('Found {} browser interactions.',
                                     len(interactions))
                        # extract events and timestamps
                        event = []
                        time = []
                        for interation in interactions:
                            if interation['trial'] == stim_trial:
                                event.append(interation['event'])
                                time.append(interation['time'])
                        # Check if inputted values were recorded previously
                        if stim_name + '-event' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-event'] = event
                        else:
                            # previous values found
                            dict_row[stim_name + '-event'].append(event)
                        # Check if time spent values were recorded
                        # previously
                        if stim_name + '-time' not in dict_row.keys():
                            # first value
                            dict_row[stim_name + '-time'] = time
                        else:
                            # previous values found
                            dict_row[stim_name + '-time'].append(time)
                # worker_code was ecnountered before
                if dict_row['worker_code'] in data_dict.keys():
                    # iterate over items in the data dictionary
                    for key, value in dict_row.items():
                        # new value
                        if key not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                            data_dict[dict_row['worker_code']][key] = value
                        # update old value
                        else:
                            # udpate only if it is a list
                            if isinstance(data_dict[dict_row['worker_code']][key], list):  # noqa: E501
                                data_dict[dict_row['worker_code']][key].extend(value)  # noqa: E501
                # worker_code is ecnountered for the first time
                else:
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into pandas dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # report people that attempted study
            unique_worker_codes = df['worker_code'].drop_duplicates()
            logger.info('People who attempted to participate: {}',
                        unique_worker_codes.shape[0])
            # filter data
            df = self.filter_data(df)
        # save to pickle
        if self.save_p:
            cs.common.save_to_p(self.file_p, df, 'heroku data')
        # save to csv
        if self.save_csv:
            # todo: check whith index=False is needed here
            df.to_csv(cs.settings.output_dir + '/' + self.file_data_csv +
                      '.csv', index=False)
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_mapping(self):
        """
        Read mapping.
        """
        # read mapping from a csv file
        mapping = pd.read_csv(cs.common.get_configs('mapping_stimuli'))
        # set index as stimulus_id
        mapping.set_index('video_id', inplace=True)
        # return mapping as a dataframe
        return mapping

    def populate_mapping(self, df, points_duration, mapping):
        """
        Populate dataframe with mapping of stimuli with counts of detected
        coords for each stimulus duration.
        """
        # todo: add analysis info to mapping matrix. for example mean keypresses
        logger.info('Populating coordinates in mapping of stimuli')
        # # read mapping of polygons from a csv file
        # polygons = pd.read_csv(cs.common.get_configs('vehicles_polygons'))
        # # set index as stimulus_id
        # polygons.set_index('image_id', inplace=True)
        # # loop over stimuli
        # for stim_id in tqdm(range(1, self.num_stimuli + 1)):
        #     # polygon of vehicle
        #     coords = np.array(polygons.at[stim_id, 'coords'].split(','),
        #                       dtype=int).reshape(-1, 2)
        #     polygon = Polygon(coords)
        #     # loop over durations of stimulus
        #     for duration in range(len(self.durations)):
        #         # loop over coord in the list of coords
        #         for point in points_duration[duration][stim_id]:
        #             # convert to point object
        #             point = Point(point[0], point[1])
        #             # check if point is within polygon of vehicle
        #             if polygon.contains(point):
        #                 # check if nan is in the cell
        #                 if pd.isna(mapping.at[stim_id,
        #                                       self.durations[duration]]):
        #                     mapping.at[stim_id, self.durations[duration]] = 1
        #                 # not nan
        #                 else:
        #                     mapping.at[stim_id, self.durations[duration]] += 1
        #         # count number of participants per duration
        #         name_cell = 'image_' + \
        #                     str(stim_id) + \
        #                     '-' + str(self.durations[duration]) + \
        #                     '-cb'
        #         count = int(self.heroku_data[name_cell].count())
        #         mapping.at[stim_id,
        #                    str(self.durations[duration]) + '_count'] = count
        #     # add area of vehicle polygon
        #     mapping.at[stim_id, 'veh_area'] = polygon.area
        # # add mean value of counts
        # mapping['gazes_mean'] = mapping[self.durations].mean(axis=1)
        # # convert counts of participants to integers
        # for duration in range(len(self.durations)):
        #     column = str(self.durations[duration]) + '_count'
        #     mapping[column] = mapping[column].astype(int)
        # save to csv
        if self.save_csv:
            # save to csv
            mapping.to_csv(cs.settings.output_dir + '/' +
                           self.file_mapping_csv + '.csv')
        # return mapping
        return mapping

    def filter_data(self, df):
        """
        Filter data based on the folllowing criteria:
            1. People who made mistakes in injected questions.
        """
        # load mapping of codes and coordinates
        logger.info('Filtering heroku data.')
        # 1. People who made mistakes in injected questions
        logger.info('Filter-h1. People who made mistakes in injected '
                    + 'questions.')
        allowed_mistakes = cs.common.get_configs('allowed_mistakes_injections')  # noqa: E501
        # injection questions
        injections = cs.common.get_configs('injections')
        # answers to injected questions
        injections_answers = cs.common.get_configs('injections_answers')
        # # df to store data to filter out
        df_1 = pd.DataFrame()
        # loop over rows in data
        # tqdm adds progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # fill nans with empty lists
            empty = pd.Series([[] for _ in range(len(row.index))],
                              index=row.index)
            row = row.fillna(empty)
            # counter mistakes
            mistakes_counter = 0
            # counter sentinel images found in training
            injections_counter = 0
            # questions detected
            questions_detected = False
            questions = []
            # answers detected
            answers_detected = False
            answers = []
            # loop over values in the row
            for index_r, value_r in row.iteritems():
                # check if input is given
                if (value_r == []):
                    # if no data present, move to the next cell
                    continue
                # questions detected
                if '-qs' in index_r:
                    questions_detected = True
                    questions = value_r
                # answers detected
                if '-as' in index_r:
                    answers_detected = True
                    answers = value_r
                # injected question
                if '-qi' in index_r:
                    # there can be multiple injections per stimulus
                    for question in value_r:
                        # counter of processed questions
                        processed_counter = 0
                        # check if it indeed an injection
                        if question != 'na':
                            # increase counter of injections
                            injections_counter = injections_counter + 1
                            # correct answer
                            index = injections.index(question)
                            correct_answer = injections_answers[index]
                            # given answer (multiple possible)
                            indices = [i for i, x in enumerate(questions) if x == "injection"]  # noqa: E501
                            index = indices[processed_counter]
                            given_answer = answers[index]
                            # increase counter of processed questions
                            processed_counter = processed_counter + 1
                            if given_answer != correct_answer:
                                # mistake found
                                mistakes_counter = mistakes_counter + 1
                                # check if limit was reached
                                if mistakes_counter > allowed_mistakes:
                                    logger.debug('{}: found {} mistakes for '
                                                 + 'question injections.',
                                                 row['worker_code'],
                                                 mistakes_counter)
                                    # add to df with data to filter out
                                    df_1 = df_1.append(row)
                                    break
        logger.info('Filter-h1. People who made more than {} mistakes with '
                    + 'injected questions: {}',
                    allowed_mistakes,
                    df_1.shape[0])
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        # people to filter present
        if df_1.shape[0] != 0:
            df_filtered = pd.concat([df_1])
            # drop rows with filtered data
            unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
            df = df[~df['worker_code'].isin(unique_worker_codes)]
        logger.info('Filtered in total in heroku data: {}',
                    old_size - df.shape[0])
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        logger.info('No info to show.')
