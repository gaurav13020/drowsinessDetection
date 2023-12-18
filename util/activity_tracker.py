from stopwatch import Stopwatch as Timer

class ActivityTracker:
    activities = dict()
    current_activity = None

    def start_activity(self, activity):
        
        # Initialize activity if not already there
        if activity not in self.activities.keys():
            self.activities[activity] = Timer()
            self.activities[activity].reset()

        # Stop previous activity if one already running and it is not the same one
        if self.current_activity is not activity:
            if self.current_activity:
                self.activities[self.current_activity].stop()
        
        self.activities[activity].start()
        self.current_activity = activity
    
    def current(self):
        return self.current_activity

    def get_activities(self):
        return self.activities.keys()

    def clear(self):
        for activity, timer in self.activities.items():
            timer.reset()

    def get_time(self, activity):
        if activity in self.activities.keys():
            return self.activities[activity].duration
    
    def get_total_time(self):
        total = 0
        for activity in self.activities.keys():
            total += self.activities[activity].duration
        return total

    def print_activities(self):
        for activity, timer in self.activities.items():
            print(activity + ": " + str(timer.duration))
        print("\n")
    
    # Returns a dictionary of activity -> times (not activity -> timer)
    def get_activities_dict(self):
        output = dict()
        for activity in self.activities.keys():
            output[activity] = self.activities[activity].duration
        return output