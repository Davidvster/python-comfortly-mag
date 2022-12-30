pa_items = ["interested", "excited", "strong", "determined", "attentive", "alert", "enthusiastic", "inspired", "active",
            "proud"]
na_items = ["afraid", "distressed", "upset", "guilty", "ashamed", "irritable", "hostile", "jittery", "nervous", "upset"]

ratings = {"Skoraj nič/nič": 1, "Malo": 2, "Zmerno": 3, "Kar veliko": 4, "Zelo": 5}


class PanasEvaluation:
    def __init__(self, answers):
        self.pa = 0
        self.na = 0
        self.answers = answers
        for answer in self.answers:
            if any(s in answer[2].lower() for s in pa_items):
                self.pa += ratings[answer[3]]
            elif any(s in answer[2].lower() for s in na_items):
                self.na += ratings[answer[3]]

    def get_pa(self):
        return self.pa

    def get_na(self):
        return self.na
