class CogniVisionScorer:
    """
    Calculates engagement metrics based on AI engine results.
    """
    def __init__(self, phone_penalty=20):
        self.phone_penalty = phone_penalty

    def calculate_class_score(self, engine_results):
        """
        Calculates the overall class attention percentage.
        Formula: (Attentive Students / Total Students) * 100 - Penalties
        """
        students = [r for r in engine_results if r['type'] == 'student']
        phones = [r for r in engine_results if r['status'] == 'distraction (phone)']
        
        if not students:
            return 0.0
        
        attentive_count = sum(1 for s in students if s['status'] == 'attentive')
        base_score = (attentive_count / len(students)) * 100
        
        # Apply penalties for distractions
        # We cap the score at 0
        final_score = max(0.0, base_score - (len(phones) * self.phone_penalty))
        
        return round(final_score, 2)

    def get_individual_report(self, engine_results):
        """
        Drafts a brief report for debugging/logs.
        """
        students = [r for r in engine_results if r['type'] == 'student']
        return {
            "total_students": len(students),
            "attentive": sum(1 for s in students if s['status'] == 'attentive'),
            "distracted": sum(1 for s in students if s['status'] == 'distracted'),
            "score": self.calculate_class_score(engine_results)
        }
