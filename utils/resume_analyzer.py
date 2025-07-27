import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import io
from typing import Dict, List, Optional
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class ResumeAnalyzer:
    """Analyzes resumes and extracts key information"""
    
    def __init__(self):
        # Common skills database
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'r', 'matlab', 'scala', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'bootstrap', 'jquery'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib', 'seaborn', 'plotly', 'tableau'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra', 'elasticsearch'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
            'analytics': ['excel', 'power bi', 'qlik', 'looker', 'spark', 'hadoop', 'kafka', 'airflow']
        }
        
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'creative', 'organized', 'detail-oriented', 'adaptable', 'collaborative',
            'project management', 'strategic thinking', 'innovation', 'mentoring'
        ]
        
        # Education levels
        self.education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'masters': ['master', 'msc', 'ms', 'mba', 'ma', 'meng'],
            'bachelors': ['bachelor', 'bsc', 'bs', 'ba', 'beng', 'btch'],
            'associates': ['associate', 'aa', 'as', 'aas'],
            'certification': ['certificate', 'certification', 'certified']
        }
        
    def extract_text_from_upload(self, uploaded_file) -> str:
        """
        Extract text from uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Extracted text content
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                # Handle text files
                return str(uploaded_file.read(), 'utf-8')
            elif file_extension == 'pdf':
                if not PDF_AVAILABLE:
                    st.error("PDF parsing libraries not available. Please upload a text file version of the resume.")
                    return ""
                
                # Handle PDF files
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Try pdfplumber first (better text extraction)
                    import pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if text.strip():
                        return text
                    
                    # Fallback to PyPDF2
                    uploaded_file.seek(0)
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    
                    return text
                    
                except Exception as pdf_error:
                    st.error(f"Error reading PDF: {str(pdf_error)}")
                    return ""
            else:
                st.error(f"Unsupported file format: {file_extension}. Supported formats: TXT, PDF")
                return ""
                
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def analyze_resume(self, text: str) -> Dict:
        """
        Analyze resume text and extract key information
        
        Args:
            text (str): Resume text content
            
        Returns:
            dict: Analysis results
        """
        text_lower = text.lower()
        
        analysis = {
            'technical_skills': self._extract_technical_skills(text_lower),
            'soft_skills': self._extract_soft_skills(text_lower),
            'education': self._extract_education(text_lower),
            'experience_years': self._estimate_experience(text),
            'contact_info': self._extract_contact_info(text),
            'certifications': self._extract_certifications(text_lower),
            'summary_stats': {}
        }
        
        # Generate summary statistics
        analysis['summary_stats'] = {
            'total_technical_skills': len(analysis['technical_skills']),
            'total_soft_skills': len(analysis['soft_skills']),
            'education_level': analysis['education'].get('highest_level', 'Unknown'),
            'estimated_experience': analysis['experience_years'],
            'skills_diversity': len(set(category for category in analysis['technical_skills'].keys() if analysis['technical_skills'][category]))
        }
        
        return analysis
    
    def _extract_technical_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical skills by category"""
        found_skills = {}
        
        for category, skills in self.technical_skills.items():
            found_in_category = []
            for skill in skills:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    found_in_category.append(skill.title())
            found_skills[category] = found_in_category
            
        return found_skills
    
    def _extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills"""
        found_skills = []
        
        for skill in self.soft_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill.title())
                
        return found_skills
    
    def _extract_education(self, text: str) -> Dict:
        """Extract education information"""
        education = {
            'degrees': [],
            'institutions': [],
            'highest_level': 'Unknown'
        }
        
        # Find degrees
        for level, keywords in self.education_levels.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    education['degrees'].append(level.title())
                    
        # Determine highest level
        if 'Phd' in education['degrees']:
            education['highest_level'] = 'PhD'
        elif 'Masters' in education['degrees']:
            education['highest_level'] = 'Masters'
        elif 'Bachelors' in education['degrees']:
            education['highest_level'] = 'Bachelors'
        elif 'Associates' in education['degrees']:
            education['highest_level'] = 'Associates'
        elif 'Certification' in education['degrees']:
            education['highest_level'] = 'Certification'
            
        # Extract university names (basic pattern matching)
        university_patterns = [
            r'university of \w+',
            r'\w+ university',
            r'\w+ college',
            r'\w+ institute'
        ]
        
        for pattern in university_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education['institutions'].extend([match.title() for match in matches])
            
        education['institutions'] = list(set(education['institutions']))  # Remove duplicates
        
        return education
    
    def _estimate_experience(self, text: str) -> int:
        """Estimate years of experience from resume text"""
        # Look for explicit experience mentions
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*:\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*professional'
        ]
        
        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([int(match) for match in matches])
            
        if years:
            return max(years)
        
        # Fallback: count job positions and estimate
        job_patterns = [
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s*-\s*(?:present|current|january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}',
            r'\d{4}\s*-\s*(?:present|current|\d{4})'
        ]
        
        job_periods = 0
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_periods += len(matches)
            
        # Rough estimate: each job period = 2 years average
        return min(job_periods * 2, 15) if job_periods > 0 else 0
    
    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information"""
        contact: Dict[str, Optional[str]] = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact['email'] = email_match.group()
            
        # Phone pattern
        phone_pattern = r'[\+]?[1-9]?[\-\.\s]?\(?[0-9]{3}\)?[\-\.\s]?[0-9]{3}[\-\.\s]?[0-9]{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group()
            
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w\-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact['linkedin'] = linkedin_match.group()
            
        # GitHub pattern
        github_pattern = r'github\.com/[\w\-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact['github'] = github_match.group()
            
        return contact
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_patterns = [
            r'certified\s+[\w\s]+',
            r'[\w\s]+\s+certified',
            r'certification\s+in\s+[\w\s]+',
            r'[\w\s]+\s+certification'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend([match.strip().title() for match in matches])
            
        # Remove duplicates and common false positives
        certifications = list(set(certifications))
        filtered_certs = [cert for cert in certifications if len(cert.split()) <= 5 and len(cert) > 5]
        
        return filtered_certs[:10]  # Limit to first 10 to avoid noise
    
    def get_suitable_jobs(self, analysis: Dict) -> List[Dict]:
        """
        Get suitable job positions based on resume analysis
        
        Args:
            analysis (dict): Resume analysis results
            
        Returns:
            list: List of suitable job positions
        """
        # Job database with requirements
        job_database = [
            {
                'title': 'Software Engineer',
                'company': 'Tech Corp',
                'requirements': ['Python', 'JavaScript', 'SQL', 'Git'],
                'experience_required': 2,
                'education_required': 'Bachelors',
                'salary_range': '$70,000 - $95,000'
            },
            {
                'title': 'Senior Software Engineer',
                'company': 'Innovation Labs',
                'requirements': ['Python', 'React', 'Node.js', 'AWS', 'Leadership'],
                'experience_required': 5,
                'education_required': 'Bachelors',
                'salary_range': '$95,000 - $130,000'
            },
            {
                'title': 'Data Scientist',
                'company': 'Analytics Inc',
                'requirements': ['Python', 'Machine Learning', 'SQL', 'TensorFlow', 'Statistics'],
                'experience_required': 3,
                'education_required': 'Masters',
                'salary_range': '$85,000 - $120,000'
            },
            {
                'title': 'Full Stack Developer',
                'company': 'StartupXYZ',
                'requirements': ['JavaScript', 'React', 'Node.js', 'MongoDB', 'CSS'],
                'experience_required': 3,
                'education_required': 'Bachelors',
                'salary_range': '$75,000 - $100,000'
            },
            {
                'title': 'DevOps Engineer',
                'company': 'Cloud Solutions',
                'requirements': ['AWS', 'Docker', 'Kubernetes', 'Jenkins', 'Python'],
                'experience_required': 4,
                'education_required': 'Bachelors',
                'salary_range': '$90,000 - $125,000'
            },
            {
                'title': 'Frontend Developer',
                'company': 'Design Studio',
                'requirements': ['JavaScript', 'React', 'CSS', 'HTML', 'UI/UX'],
                'experience_required': 2,
                'education_required': 'Bachelors',
                'salary_range': '$65,000 - $85,000'
            },
            {
                'title': 'Product Manager',
                'company': 'Product Co',
                'requirements': ['Leadership', 'Project Management', 'Analytics', 'Communication'],
                'experience_required': 5,
                'education_required': 'Bachelors',
                'salary_range': '$100,000 - $140,000'
            },
            {
                'title': 'Backend Developer',
                'company': 'Server Solutions',
                'requirements': ['Python', 'Java', 'SQL', 'REST API', 'Database Design'],
                'experience_required': 3,
                'education_required': 'Bachelors',
                'salary_range': '$80,000 - $110,000'
            }
        ]
        
        # Get candidate's skills and experience
        all_candidate_skills = []
        for category_skills in analysis['technical_skills'].values():
            all_candidate_skills.extend([skill.lower() for skill in category_skills])
        all_candidate_skills.extend([skill.lower() for skill in analysis['soft_skills']])
        
        candidate_experience = analysis['summary_stats']['estimated_experience']
        candidate_education = analysis['summary_stats']['education_level']
        
        # Education level mapping
        education_levels = {
            'PhD': 4,
            'Masters': 3,
            'Bachelors': 2,
            'Associates': 1,
            'Certification': 0
        }
        
        candidate_edu_level = education_levels.get(candidate_education, 0)
        suitable_jobs = []
        
        for job in job_database:
            # Calculate skill match
            job_requirements_lower = [req.lower() for req in job['requirements']]
            matched_skills = [skill for skill in job_requirements_lower if skill in all_candidate_skills]
            missing_skills = [skill for skill in job_requirements_lower if skill not in all_candidate_skills]
            
            skill_match_percentage = (len(matched_skills) / len(job_requirements_lower)) * 100
            
            # Check experience requirement
            experience_match = candidate_experience >= job['experience_required']
            
            # Check education requirement
            required_edu_level = education_levels.get(job['education_required'], 2)
            education_match = candidate_edu_level >= required_edu_level
            
            # Calculate overall match score
            match_score = skill_match_percentage
            
            # Bonus points for meeting experience and education requirements
            if experience_match:
                match_score += 10
            if education_match:
                match_score += 10
                
            # Penalty for not meeting basic requirements
            if not experience_match:
                match_score -= 15
            if not education_match:
                match_score -= 20
            
            match_score = max(0, min(100, match_score))  # Keep between 0-100
            
            # Only include jobs with reasonable match (at least 30% or meets basic requirements)
            if match_score >= 30 or (skill_match_percentage >= 50 and (experience_match or education_match)):
                # Generate match reason
                match_reason = []
                if skill_match_percentage >= 70:
                    match_reason.append("Strong skill alignment")
                elif skill_match_percentage >= 50:
                    match_reason.append("Good skill match")
                else:
                    match_reason.append("Partial skill match")
                
                if experience_match:
                    match_reason.append("meets experience requirement")
                else:
                    match_reason.append(f"needs {job['experience_required'] - candidate_experience} more years experience")
                
                if education_match:
                    match_reason.append("meets education requirement")
                else:
                    match_reason.append(f"consider pursuing {job['education_required']} degree")
                
                suitable_jobs.append({
                    'title': job['title'],
                    'company': job['company'],
                    'requirements': job['requirements'],
                    'experience_required': job['experience_required'],
                    'education_required': job['education_required'],
                    'salary_range': job['salary_range'],
                    'match_score': round(match_score, 1),
                    'matched_skills': matched_skills,
                    'missing_skills': missing_skills,
                    'match_reason': ', '.join(match_reason)
                })
        
        # Sort by match score (descending)
        suitable_jobs.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Return top 5 matches
        return suitable_jobs[:5]
    
    def generate_skill_match_score(self, resume_skills: Dict, job_requirements: List[str]) -> Dict:
        """
        Calculate skill match score against job requirements
        
        Args:
            resume_skills (dict): Extracted resume skills
            job_requirements (list): List of required skills
            
        Returns:
            dict: Match analysis
        """
        # Flatten all technical skills from resume
        all_resume_skills = []
        for category_skills in resume_skills['technical_skills'].values():
            all_resume_skills.extend([skill.lower() for skill in category_skills])
        
        all_resume_skills.extend([skill.lower() for skill in resume_skills['soft_skills']])
        
        # Calculate matches
        job_requirements_lower = [req.lower() for req in job_requirements]
        matched_skills = [skill for skill in job_requirements_lower if skill in all_resume_skills]
        missing_skills = [skill for skill in job_requirements_lower if skill not in all_resume_skills]
        
        match_score = (len(matched_skills) / len(job_requirements_lower)) * 100 if job_requirements_lower else 0
        
        return {
            'match_score': round(match_score, 1),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'total_requirements': len(job_requirements_lower),
            'total_matches': len(matched_skills)
        }
    
    def create_skills_visualization_data(self, analysis: Dict) -> pd.DataFrame:
        """
        Create data for skills visualization
        
        Args:
            analysis (dict): Resume analysis results
            
        Returns:
            pandas.DataFrame: Skills data for visualization
        """
        skills_data = []
        
        # Technical skills by category
        for category, skills in analysis['technical_skills'].items():
            for skill in skills:
                skills_data.append({
                    'Skill': skill,
                    'Category': category.replace('_', ' ').title(),
                    'Type': 'Technical',
                    'Count': 1
                })
        
        # Soft skills
        for skill in analysis['soft_skills']:
            skills_data.append({
                'Skill': skill,
                'Category': 'Soft Skills',
                'Type': 'Soft',
                'Count': 1
            })
        
        return pd.DataFrame(skills_data)