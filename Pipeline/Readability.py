import math
class ReadabilityDetails:
    
    @staticmethod
    def ari(score):
        score = math.ceil(score)
        if score <= 1:
            s = ['K']
        elif score <= 2:
            s = ['1', '2']
        elif score <= 3:
            s = ['3']
        elif score <= 4:
            s = ['4']
        elif score <= 5:
            s = ['5']
        elif score <= 6:
            s = ['6']
        elif score <= 7:
            s = ['7']
        elif score <= 8:
            s = ['8']
        elif score <= 9:
            s = ['9']
        elif score <= 10:
            s = ['10']
        elif score <= 11:
            s = ['11']
        elif score <= 12:
            s = ['12']
        elif score <= 13:
            s = ['college']
        else:
            s = ['college_graduate']

        score = math.ceil(score)
        if score <= 1:
            a = [5, 6]
        elif score <= 2:
            a = [6, 7]
        elif score <= 3:
            a = [7, 9]
        elif score <= 4:
            a = [9, 10]
        elif score <= 5:
            a = [10, 11]
        elif score <= 6:
            a = [11, 12]
        elif score <= 7:
            a = [12, 13]
        elif score <= 8:
            a = [13, 14]
        elif score <= 9:
            a = [14, 15]
        elif score <= 10:
            a = [15, 16]
        elif score <= 11:
            a = [16, 17]
        elif score <= 12:
            a = [17, 18]
        elif score <= 13:
            a = [18, 24]
        else:
            a = [24, 100]

        return {'grade': s, 'ages': a}

    @staticmethod
    def coleman_liau(score):
        return {'grade': [round(score)]}
    
    @staticmethod
    def dale_chall(score):
        if score <= 4.9:
            g = ['1', '2', '3', '4']
        elif score >= 5 and score < 6:
            g = ['5', '6']
        elif score >= 6 and score < 7:
            g = ['7', '8']
        elif score >= 7 and score < 8:
            g = ['9', '10']
        elif score >= 8 and score < 9:
            g = ['11', '12']
        elif score >= 9 and score < 10:
            g = ['college']
        else:
            g = ['college_graduate']

        return {'grade': g}
    
    @staticmethod
    def flesch_kincaid(score):
        return {'grade': [round(score)]}
    
    @staticmethod
    def flesch(score):
        if score >= 90 and score <= 100:
            e = 'very_easy'
        elif score >= 80 and score < 90:
            e = 'easy'
        elif score >= 70 and score < 80:
            e = 'fairly_easy'
        elif score >= 60 and score < 70:
            e = 'standard'
        elif score >= 50 and score < 60:
            e = 'fairly_difficult'
        elif score >= 30 and score < 50:
            e = 'difficult'
        else:
            e = 'very_confusing'

        if score >= 90 and score <= 100:
            g = ['5']
        elif score >= 80 and score < 90:
            g = ['6']
        elif score >= 70 and score < 80:
            g = ['7']
        elif score >= 60 and score < 70:
            g = ['8', '9']
        elif score >= 50 and score < 60:
            g = ['10', '11', '12']
        elif score >= 30 and score < 50:
            g = ['college']
        else:
            g = ['college_graduate']

        return {'grade': g, 'ease': e}
    
    @staticmethod
    def gunning_fog(score):
        rounded = round(score)
        if rounded < 6:
            g = 'na'
        elif rounded >= 6 and rounded <= 12:
            g = str(rounded)
        elif rounded >= 13 and rounded <= 16:
            g = 'college'
        else:
            g = 'college_graduate'

        return {'grade': [g]}
    
    @staticmethod
    def smog(score):
        return {'grade': [round(score)]}
    

    @staticmethod
    def averageToUK(grade):
        g = math.ceil(grade)
        ukGrade = g + 1
        lowerAge = ukGrade+4
        if lowerAge <= 4:
            year = "Reception"
            ageString = str(1) + "-" + str(4)
        elif lowerAge <= 15:
            year = "Year " + str(ukGrade)
            ageString = str(lowerAge) + "-" + str(lowerAge+1)
        elif lowerAge <= 17:
            year = "College"
            ageString = str(16) + "-" + str(18)
        elif lowerAge <= 21:
            year = "University"
            ageString = str(18) + "-" + str(24)
        else:
            year = "Postgrad" 
            ageString = str(24) + "+"

        
        return {'grade': ukGrade, 'age': ageString, 'year': year }


    
