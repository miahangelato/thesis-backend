from django.db import models
import uuid
import random
import os


def get_filename_ext(filepath):
    base_name = os.path.basename(filepath)
    name, ext = os.path.splitext(base_name)
    return name, ext

def upload_image_path(instance, filename):
    """
    Generate upload path: fingerprint_images/participant_[ID]/[finger]_[uuid].[ext]
    This ensures all 10 fingerprints for a participant are in one folder
    """
    name, ext = get_filename_ext(filename)
    # Generate unique filename using finger type and UUID
    unique_id = str(uuid.uuid4())[:8]  # Short UUID for cleaner filenames
    final_filename = f"{instance.finger}_{unique_id}{ext}"
    
    # All fingerprints for a participant go in one folder
    return f"fingerprint_images/participant_{instance.participant.id}/{final_filename}"
class Participant(models.Model):
    age = models.IntegerField()
    gender = models.CharField(
        max_length=10, 
        choices=[("male", "Male"), ("female", "Female")]
    )
    blood_type = models.CharField(
        max_length=10,  # Fix: was max_length=None which causes errors
        choices=[("A", "A"), ("B", "B"), ("AB", "AB"), ("O", "O"), ("unknown", "Unknown")],
        default="unknown"
    )
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    
    willing_to_donate = models.BooleanField(default=False)

    # Health & consent
    has_chronic_condition = models.BooleanField(null=True, default=False)
    condition_controlled = models.BooleanField(null=True, default=False)  # ✅ new
    consent = models.BooleanField(default=False)

    # Lifestyle / donation restrictions
    recent_tattoo_or_piercing = models.BooleanField(null=True, default=False)
    ate_fatty_food = models.BooleanField(null=True,default=False)
    ate_before_donation = models.BooleanField(null=True,default=True)
    had_alcohol_last_24h = models.BooleanField(null=True, default=False)
    sleep_hours = models.IntegerField(null=True, blank=True, default=None)
    last_donation_date = models.DateField(null=True, blank=True)

    def check_prc_eligibility(self):
        """Apply Philippine Red Cross blood donation criteria"""
        reasons = []
        eligible = True

        # Age
        if not (16 <= self.age <= 65):
            eligible = False
            reasons.append("Age not within 16-65 years")

        # Weight
        if self.weight and self.weight < 50:
            eligible = False
            reasons.append("Weight below 50kg")

        # Chronic condition
        if self.has_chronic_condition and not self.condition_controlled:
            eligible = False
            reasons.append("Uncontrolled chronic condition")

        # Lifestyle
        if self.recent_tattoo_or_piercing:
            eligible = False
            reasons.append("Recent tattoo/piercing")

        if self.ate_fatty_food:
            eligible = False
            reasons.append("Ate fatty food")

        if self.had_alcohol_last_24h:
            eligible = False
            reasons.append("Had alcohol in last 24 hours")

        if self.sleep_hours < 6:
            eligible = False
            reasons.append("Less than 6 hours sleep")

        # Last donation
        if self.last_donation_date:
            from datetime import timedelta, date
            if (date.today() - self.last_donation_date) < timedelta(days=90):
                eligible = False
                reasons.append("Last donation less than 3 months ago")

        # Consent
        if not self.consented:
            eligible = False
            reasons.append("No consent given")

        return eligible, reasons



class Fingerprint(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="fingerprints")
    finger = models.CharField(max_length=20)  # e.g., "left_thumb"
    image = models.ImageField(upload_to=upload_image_path)
    pattern = models.CharField(max_length=20, null=True, blank=True)  # Arc, Loop, Whorl

class Result(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="results")
    diabetes_risk = models.CharField(max_length=20)  # "low", "medium", "high"
    confidence_score = models.FloatField(null=True, blank=True)

    blood_group = models.CharField(max_length=3, choices=[("A+", "A+"), ("B+", "B+"), ("AB+", "AB+"), ("O+", "O+")])
    blood_group_confidence_score = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)


class Session(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    participant_data = models.JSONField()  # Store participant data
    diabetes_result = models.JSONField(null=True, blank=True)  # Store diabetes prediction result
    blood_group_result = models.JSONField(null=True, blank=True)  # Store blood group prediction result
    fingerprints = models.JSONField(null=True, blank=True)  # Store fingerprint data
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()  # Session expiration
    
    def __str__(self):
        return f"Session {self.session_id}"