class TrainSystem():
    def __init__(self) -> None:
        # key: start_station-end_station, value: list of time
        self.travelmap: dict = dict()
        self.checkinmap: dict = dict()

    def getAverageTime(self, start_station: str, end_station: str):
        pass

    def checkIn(self, id: int, station: str, time: int):
        self.checkinmap[id] = (station, time)

    def checkOut(self, id: int, station: str, time: int):
        # error handling
        if id not in self.checkinmap:
            raise ValueError("id not found")

        start_station, start_time = self.checkinmap[id]
        key = start_station + "-" + station
        if key not in self.travelmap:
            self.travelmap[key] = []
        self.travelmap[key].append(time - start_time)


if __name__ == "__main__":
    train = TrainSystem()
    train.checkIn(45, "Leyton", 3)
    train.checkOut(45, "Paradise", 8)
    print(train.getAverageTime("Leyton", "Paradise"))  # return 5.0
