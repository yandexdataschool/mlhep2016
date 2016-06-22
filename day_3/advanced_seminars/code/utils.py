__author__ = 'mikhail91'

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import cm

def straight_tracks_generator(n_events, n_tracks, n_noise, sigma, intersection=True, x_range=(0, 10, 1), y_range=(-30, 30, 1),  k_range=(-2, 2, 0.1), b_range=(-10, 10, 0.1)):
    """
    This function generates events with straight tracks and noise.
    :param n_events: int, number of generated events.
    :param n_tracks: int, number of generated tracks in each event. Tracks will have TrackIDs in range [0, inf).
    :param n_noise: int, number of generated random noise hits. Noise hits will have TrackID -1.
    :param sigma: float, track's hit generated with error which has normal distribution.
                  Sigma is parameter of the distribution.
    :param intersection: booleen, if False the tracks will not intersect.
    :param x_range: tuple (min, max, step), range of x values of the hits.
    :param y_range: tuple (min, max, step), range of y values of the hits. Only for intersection=False.
    :param k_range: tuple (min, max, step), range of k values of the track. y = b + k * x.
    :param b_range: tuple (min, max, step), range of b values of the track. y = b + k * x.
    :return: pandas.DataFrame
    """

    list_of_events = []

    for event_id in range(n_events):

        event_tracks = []

        # Add track hits
        if intersection:

            for track_id in range(n_tracks):


                X = numpy.arange(*x_range).reshape((-1, 1))
                k = numpy.random.choice(numpy.arange(*k_range), 1)[0]
                b = numpy.random.choice(numpy.arange(*b_range), 1)[0]
                e = numpy.random.normal(scale=sigma, size=len(X)).reshape((-1, 1))
                # y = b + k * x + e
                y = b + k * X + e

                track = numpy.concatenate(([[event_id]]*len(X),
                                           [[track_id]]*len(X),
                                           X, y), axis=1)
                event_tracks.append(track)

        else:

            y = numpy.arange(*y_range)
            y_start = numpy.random.choice(y, n_tracks, replace=False)
            y_start = numpy.sort(y_start)
            y_end = numpy.random.choice(y, n_tracks, replace=False)
            y_end = numpy.sort(y_end)
            X = numpy.arange(*x_range).reshape((-1, 1))
            delta_x = X.max() - X.min()

            for track_id in range(n_tracks):


                X = numpy.arange(*x_range).reshape((-1, 1))
                k = 1. * (y_end[track_id] - y_start[track_id]) / delta_x
                b = y_start[track_id] - k * X.min()
                e = numpy.random.normal(scale=sigma, size=len(X)).reshape((-1, 1))
                # y = b + k * x + e
                y = b + k * X + e

                track = numpy.concatenate(([[event_id]]*len(X),
                                           [[track_id]]*len(X),
                                           X, y), axis=1)
                event_tracks.append(track)



        # Add noise hits
        if n_noise > 0:
            X = numpy.random.choice(numpy.arange(*x_range), n_noise).reshape((-1, 1))
            k = numpy.random.choice(numpy.arange(*k_range), n_noise).reshape(-1,1)
            b = numpy.random.choice(numpy.arange(*b_range), n_noise).reshape(-1,1)
            y = b + k * X
            noise = numpy.concatenate(([[event_id]]*len(X),
                                       [[-1]]*len(X),
                                       X, y), axis=1)
            event_tracks.append(noise)


        event = numpy.concatenate(tuple(event_tracks), axis=0)
        list_of_events.append(event)


    all_events = numpy.concatenate(tuple(list_of_events), axis=0)
    data = pandas.DataFrame(columns=['EventID', 'TrackID', 'X', 'y'], data=all_events)

    return data

def plot_straight_tracks(event, labels=None):
    """
    Generate plot of the event with its tracks and noise hits.
    :param event: pandas.DataFrame with one event.
    :param labels: numpy.array shape=[n_hits], labels of recognized tracks.
    :return: matplotlib.pyplot object.
    """

    plt.figure(figsize=(10, 7))

    tracks_id = numpy.unique(event.TrackID.values)
    event_id = event.EventID.values[0]

    color=cm.rainbow(numpy.linspace(0,1,len(tracks_id)))

    # Plot hits
    for num, track in enumerate(tracks_id):

        X = event[event.TrackID == track].X.values.reshape((-1, 1))
        y = event[event.TrackID == track].y.values

        plt.scatter(X, y, color=color[num])

        # Plot tracks
        if track != -1:

            lr = LinearRegression()
            lr.fit(X, y)

            plt.plot(X, lr.predict(X), label=str(track), color=color[num])


    if labels != None:

        unique_labels = numpy.unique(labels)

        for lab in unique_labels:

            if lab != -1:

                X = event[labels == lab].X.values.reshape((-1, 1))
                y = event[labels == lab].y.values

                lr = LinearRegression()
                lr.fit(X, y)

                X = event.X.values.reshape((-1, 1))

                plt.plot(X, lr.predict(X), color='0', alpha=0.5)




    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.title('EventID is ' + str(event_id))
    #plt.show()



def get_circle(x, radius):

    radius2 = radius**2

    y_circle = numpy.sqrt(radius2 - (x - radius)**2)


    return y_circle

def circle_tracks_generator(n_events, n_tracks, n_noise, sigma, x_range=(0, 10, 1), radius_range=(10, 20, 1)):
    """
    This function generates events with circle tracks (x - r)**2 + y**2 = r**2 and noise.
    :param n_events: int, number of generated events.
    :param n_tracks: int, number of generated tracks in each event. Tracks will have TrackIDs in range [0, inf).
    :param n_noise: int, number of generated random noise hits. Noise hits will have TrackID -1.
    :param sigma: float, track's hit generated with error which has normal distribution.
                  Sigma is parameter of the distribution.
    :param intersection: booleen, if False the tracks will not intersect.
    :param x_range: tuple (min, max, step), range of x values of the hits.
    :param radius_range: tuple (min, max, step), range of radius values of the tracks. (x - r)**2 + y**2 = r**2
    :return: pandas.DataFrame
    """

    list_of_events = []

    for event_id in range(n_events):

        event_tracks = []

        # Add track hits
        x = numpy.arange(*x_range)
        radiuses = numpy.random.choice(numpy.arange(*radius_range), n_tracks, replace=False)



        for track_id, radius in enumerate(radiuses):


            y_circle = get_circle(x, radius)
            e = numpy.random.normal(scale=sigma, size=len(y_circle))
            y_circle = y_circle + e

            track = numpy.concatenate(([[event_id]]*len(x),
                                       [[track_id]]*len(x),
                                       x.reshape(-1, 1), y_circle.reshape(-1, 1)), axis=1)
            event_tracks.append(track)



        # Add noise hits
        if n_noise > 0:
            x = numpy.random.choice(numpy.arange(*x_range), n_noise).reshape((-1, 1))
            y = numpy.random.choice(numpy.arange(*radius_range), n_noise).reshape((-1, 1))
            noise = numpy.concatenate(([[event_id]]*len(x),
                                       [[-1]]*len(x),
                                       x, y), axis=1)
            event_tracks.append(noise)


        event = numpy.concatenate(tuple(event_tracks), axis=0)
        list_of_events.append(event)


    all_events = numpy.concatenate(tuple(list_of_events), axis=0)
    data = pandas.DataFrame(columns=['EventID', 'TrackID', 'X', 'y'], data=all_events)

    return data


def plot_circle_tracks(event, labels=None):
    """
    Generate plot of the event with its tracks and noise hits.
    :param event: pandas.DataFrame with one event.
    :param labels: numpy.array shape=[n_hits], labels of recognized tracks.
    :return: matplotlib.pyplot object.
    """

    plt.figure(figsize=(10, 7))

    tracks_id = numpy.unique(event.TrackID.values)
    event_id = event.EventID.values[0]

    color=cm.rainbow(numpy.linspace(0,1,len(tracks_id)))

    # Plot hits
    for num, track in enumerate(tracks_id):

        X = event[event.TrackID == track].X.values.reshape((-1, 1))
        y = event[event.TrackID == track].y.values

        plt.scatter(X, y, color=color[num])


    if labels != None:

        unique_labels = numpy.unique(labels)

        for lab in unique_labels:

            if lab != -1:

                X = event[labels == lab].X.values.reshape((-1, 1))
                y = event[labels == lab].y.values

                plt.plot(X, y, color='0', alpha=0.5)




    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.title('EventID is ' + str(event_id))
    #plt.show()

