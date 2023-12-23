"""
This is the class that handles the data that is output from the Delsys Trigno Base.
Create an instance of this and pass it a reference to the Trigno base for initialization.
See CollectDataController.py for a usage example.
"""
import numpy as np


class DataKernel:
    def __init__(self, trigno_base):
        self.TrigBase = trigno_base
        self.packetCount = 0
        self.sampleCount = 0
        self.allcollectiondata = [[]]
        self.channel1time = []

    def processData(self, data_queue):
        """
        Processes the data from the DelsysAPI and places it in the data_queue argument.

        :param data_queue: A queue (list) where processed data will be appended.
        """

        # Retrieve data from DelsysAPI using the GetData method.
        outArr = self.GetData()

        # Check if there is any data received.
        if outArr is not None:
            # Iterate over each channel's data in the output array.
            for i in range(len(outArr)):
                # Extend the internal allcollectiondata list for the current channel with the new data.
                # This adds the current chunk of data to the ongoing collection.
                self.allcollectiondata[i].extend(outArr[i][0].tolist())

            try:
                # Iterate over each data point in the first channel's data.
                for i in range(len(outArr[0])):
                    # Check if the data is one-dimensional.
                    if np.asarray(outArr[0]).ndim == 1:
                        # If it is, append the entire data chunk as a list to the data_queue.
                        data_queue.append(list(np.asarray(outArr, dtype="object")[0]))
                    else:
                        # If the data is multi-dimensional, append each data point across channels to the data_queue.
                        data_queue.append(
                            list(np.asarray(outArr, dtype="object")[:, i])
                        )

                try:
                    # Update packetCount and sampleCount.
                    # packetCount is incremented by the number of data points in the first channel.
                    # sampleCount is incremented by the number of samples in the first data point of the first channel.
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    # Exception handling to avoid crashes in case of unexpected data structures.
                    pass
            except IndexError:
                # Exception handling for index errors, which might occur if outArr is empty or structured unexpectedly.
                pass

    def processYTData(self, data_queue):
        """Processes the data from the DelsysAPI and place it in the data_queue argument"""
        outArr = self.GetYTData()
        if outArr is not None:
            for i in range(len(outArr)):
                self.allcollectiondata[i].extend(outArr[i][0].tolist())
            try:
                yt_outArr = []
                for i in range(len(outArr)):
                    chan_yt = outArr[i]
                    chan_ydata = np.asarray(
                        [k.Item2 for k in chan_yt[0]], dtype="object"
                    )
                    yt_outArr.append(chan_ydata)

                data_queue.append(list(yt_outArr))

                try:
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    pass
            except IndexError:
                pass

    def GetData(self):
        """Check if data ready from DelsysAPI via Aero CheckDataQueue() - Return True if data is ready
        Get data (PollData)
        Organize output channels by their GUID keys

        Return array of all channel data
        """

        dataReady = (
            self.TrigBase.CheckDataQueue()
        )  # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = (
                self.TrigBase.PollData()
            )  # Dictionary<Guid, List<double>> (key = Guid (Unique channel ID), value = List(Y) (Y = sample value)
            outArr = [
                [] for i in range(len(DataOut.Keys))
            ]  # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(
                DataOut.Keys
            )  # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):  # loop all channels
                chan_data = DataOut[
                    channel_guid_keys[j]
                ]  # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(
                    np.asarray(chan_data, dtype="object")
                )  # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None

    def GetYTData(self):
        """YT Data stream only available when passing 'True' to Aero Start() command i.e. TrigBase.Start(True)
        Check if data ready from DelsysAPI via Aero CheckYTDataQueue() - Return True if data is ready
        Get data (PollYTData)
        Organize output channels by their GUID keys

        Return array of all channel data
        """

        dataReady = (
            self.TrigBase.CheckYTDataQueue()
        )  # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = (
                self.TrigBase.PollYTData()
            )  # Dictionary<Guid, List<(double, double)>> (key = Guid (Unique channel ID), value = List<(T, Y)> (T = time stamp in seconds Y = sample value)
            outArr = [
                [] for i in range(len(DataOut.Keys))
            ]  # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(
                DataOut.Keys
            )  # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):  # loop all channels
                chan_yt_data = DataOut[
                    channel_guid_keys[j]
                ]  # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(
                    np.asarray(chan_yt_data, dtype="object")
                )  # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None
