import streamlit as st
import nltk
import time
import numpy as np
import matplotlib.pyplot as plt
import io

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Text Style Transfer",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .text-box {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .modern-box {
        border-left: 5px solid #4338ca;
    }
    .classic-box {
        border-left: 5px solid #0ea5e9;
    }
    .stProgress > div > div > div > div {
        background-color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)

# Simulate style transfer from modern to classic
def modern_to_classic(text, temperature=0.8):
    # Simulate processing time
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
    
    # Extract key phrases (simplified for simulation)
    sentences = nltk.sent_tokenize(text)
    key_phrase = sentences[0] if sentences else text
    
    # Simulate style transfer
    if "artificial intelligence" in text.lower() or "ai" in text.lower():
        return f"Lo, the mechanical minds of which men speak, these artifices of calculation and reason, doth present a most curious prospect for mankind's future endeavors. As the ancient philosophers did contemplate the nature of thought itself, so too must we ponder these new creations born of human ingenuity.\n\nPerchance these devices shall bring forth a golden age, wherein the burdens of labor are lifted from mortal shoulders. Yet one must also consider, with grave solemnity, whether such creations might, like Prometheus unbound, bring forth consequences unforeseen by their creators."
    
    elif "climate" in text.lower() or "environment" in text.lower():
        return f"The great firmament above and the verdant earth below have, since time immemorial, existed in delicate harmony. Yet in these latter days, a most troubling change hath been observed by learned men who study the elements.\n\nThe very air grows warmer with each passing season, and the ancient rhythms of nature fall into disarray. The wise kings of distant lands have gathered their counselors to speak of this matter, though many common folk remain ignorant of the gathering storm. It behooves all men of good conscience to consider how their actions might affect the natural world."
    
    elif "technology" in text.lower() or "digital" in text.lower():
        return f"In this age of wonders, man hath created devices of such intricate design as would cause the ancients to marvel in disbelief. These mechanisms, wrought not of brass and steam but of invisible forces and crystalline materials, have transformed the very fabric of society.\n\nWhere once a message might take fortnight to travel betwixt distant cities, now words and images fly through the ether in the blink of an eye. Yet one must ponder whether, in our haste to embrace these modern miracles, we have sacrificed something of our essential nature."
    
    else:
        # Generic transformation
        return f"It hath come to the attention of this humble observer that {key_phrase.lower()} presents a matter of great consequence for all who dwell in these times.\n\nOne is reminded of the ancient tale wherein a wise king, upon his deathbed, gathered his three sons to impart his final wisdom. 'My children,' said he, 'seek not the transient pleasures of earthly existence, but rather the eternal truths that lie hidden beneath the veil of ordinary perception.'\n\nThus it has ever been with matters of such profound importance, and thus shall it remain until the final trumpet sounds."

# Simulate style transfer from classic to modern
def classic_to_modern(text, temperature=0.8):
    # Simulate processing time
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
    
    # Extract key phrases (simplified for simulation)
    sentences = nltk.sent_tokenize(text)
    key_phrase = sentences[0] if sentences else text
    
    # Simulate style transfer
    if "best of times" in text.lower() or "worst of times" in text.lower():
        return f"Economic inequality has reached unprecedented levels according to recent studies. Data from the World Economic Forum shows that the gap between the wealthy and the poor continues to widen in most developed nations.\n\nResearchers at Stanford University found that this polarization creates a 'tale of two economies' where different segments of society experience dramatically different economic realities. A 2023 survey revealed that 68% of respondents believe this economic divide is the defining challenge of our generation."
    
    elif "happy families" in text.lower() or "unhappy" in text.lower():
        return f"Recent psychological research suggests that functional relationships share common patterns, while dysfunctional relationships tend to break down in unique ways. A longitudinal study tracking 500 couples over 15 years found that successful relationships consistently demonstrated effective communication, emotional regulation, and conflict resolution strategies.\n\nIn contrast, relationships that ultimately failed showed highly individualized patterns of dysfunction. Dr. Sarah Johnson, lead researcher at the Family Dynamics Institute, notes that 'while healthy relationships look remarkably similar, troubled relationships each have their own specific issues that require tailored interventions.'"
    
    elif "to be or not to be" in text.lower():
        return f"The existential questions that humans face in decision-making have been extensively studied by neuroscientists and psychologists. Recent fMRI studies show that contemplating major life choices activates both the analytical prefrontal cortex and emotional centers like the amygdala.\n\nA 2022 paper published in Nature Neuroscience found that when people face significant crossroads in their lives, they experience a measurable neural conflict between risk-aversion and opportunity-seeking brain functions. This research helps explain why major decisions often create such psychological tension."
    
    else:
        # Generic transformation
        return f"Recent analysis of {key_phrase.lower()} reveals significant implications for contemporary society. According to data collected from multiple sources, this phenomenon affects approximately 65% of the population in developed countries.\n\nExperts in the field suggest that understanding these patterns could lead to more effective policy decisions and improved outcomes. A 2023 survey conducted by researchers at MIT found that 72% of respondents considered this issue 'very important' or 'extremely important' to their daily lives.\n\nFurther research is needed to fully understand the long-term implications, but preliminary findings indicate promising directions for future investigation."

# Function to compare text characteristics
def compare_texts(text1, text2):
    # Analyze both texts
    def analyze_text(text):
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        
        # Remove punctuation from words
        words = [word for word in words if word.isalnum()]
        
        # Calculate metrics
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(1, word_count)
        avg_sentence_length = word_count / max(1, len(sentences))
        
        return {
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "unique_words": unique_words,
            "lexical_diversity": lexical_diversity,
            "avg_sentence_length": avg_sentence_length,
            "sentence_count": len(sentences)
        }
    
    # Analyze both texts
    metrics1 = analyze_text(text1)
    metrics2 = analyze_text(text2)
    
    # Create comparison visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Select metrics to compare
    metrics_to_plot = ['word_count', 'avg_word_length', 'unique_words', 'lexical_diversity', 'avg_sentence_length']
    labels = ['Word Count', 'Avg Word Length', 'Unique Words', 'Lexical Diversity', 'Avg Sentence Length']
    
    # Normalize values for better visualization
    values1 = [metrics1[m] for m in metrics_to_plot]
    values2 = [metrics2[m] for m in metrics_to_plot]
    
    # Find max values for normalization
    max_values = [max(v1, v2) for v1, v2 in zip(values1, values2)]
    norm_values1 = [v/max_v if max_v > 0 else 0 for v, max_v in zip(values1, max_values)]
    norm_values2 = [v/max_v if max_v > 0 else 0 for v, max_v in zip(values2, max_values)]
    
    # Plot the normalized data
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, norm_values1, width, label='Original', color='#4338ca', alpha=0.7)
    ax.bar(x + width/2, norm_values2, width, label='Transformed', color='#0ea5e9', alpha=0.7)
    
    ax.set_ylabel('Normalized Value')
    ax.set_title('Text Characteristics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    ax.legend()
    
    fig.tight_layout()
    
    # Add annotations with actual values
    for i, v in enumerate(values1):
        if i == 3:  # Format lexical diversity differently
            ax.text(i - width/2, norm_values1[i] + 0.05, f"{v:.2f}", 
                    ha='center', va='bottom', fontsize=9)
        else:
            ax.text(i - width/2, norm_values1[i] + 0.05, f"{v:.1f}", 
                    ha='center', va='bottom', fontsize=9)
            
    for i, v in enumerate(values2):
        if i == 3:  # Format lexical diversity differently
            ax.text(i + width/2, norm_values2[i] + 0.05, f"{v:.2f}", 
                    ha='center', va='bottom', fontsize=9)
        else:
            ax.text(i + width/2, norm_values2[i] + 0.05, f"{v:.1f}", 
                    ha='center', va='bottom', fontsize=9)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Main app
def main():
    st.title("🔄 Text Style Transfer")
    st.markdown("""
    This app demonstrates style transfer between modern and classic text styles.
    Enter text in one style and see it transformed to the other style!
    """)
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("Style Transfer Settings")
    
    transfer_direction = st.sidebar.radio(
        "Transfer Direction",
        ["Modern → Classic", "Classic → Modern"]
    )
    
    temperature = st.sidebar.slider(
        "Temperature (Creativity)",
        0.1, 1.5, 0.8,
        help="Higher values make output more creative/diverse"
    )
    
    # Sample text examples
    if transfer_direction == "Modern → Classic":
        sample_texts = [
            "Select a sample text...",
            "Artificial intelligence is transforming how we interact with technology.",
            "The data shows that climate change is accelerating faster than predicted.",
            "Digital platforms have revolutionized how we communicate and share information."
        ]
    else:
        sample_texts = [
            "Select a sample text...",
            "It was the best of times, it was the worst of times.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "To be or not to be, that is the question."
        ]
    
    selected_sample = st.sidebar.selectbox("Sample Texts", sample_texts)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        if transfer_direction == "Modern → Classic":
            st.header("Modern Text (Input)")
            style_class = "modern-box"
        else:
            st.header("Classic Text (Input)")
            style_class = "classic-box"
        
        # Text input
        if selected_sample == "Select a sample text...":
            if transfer_direction == "Modern → Classic":
                default_text = "Artificial intelligence is transforming how we interact with technology."
            else:
                default_text = "It was the best of times, it was the worst of times."
        else:
            default_text = selected_sample
            
        input_text = st.text_area("Enter text to transform:", default_text, height=200)
    
    with col2:
        if transfer_direction == "Modern → Classic":
            st.header("Classic Text (Output)")
        else:
            st.header("Modern Text (Output)")
    
    # Transform button
    if st.button("Transform Text", type="primary"):
        with col2:
            with st.spinner("Transforming text..."):
                if transfer_direction == "Modern → Classic":
                    transformed_text = modern_to_classic(input_text, temperature)
                    output_style_class = "classic-box"
                else:
                    transformed_text = classic_to_modern(input_text, temperature)
                    output_style_class = "modern-box"
                
                st.markdown(f"<div class='text-box {output_style_class}'>{transformed_text}</div>", unsafe_allow_html=True)
                
                # Store for comparison
                st.session_state.input_text = input_text
                st.session_state.transformed_text = transformed_text
        
        # Show comparison
        st.markdown("---")
        st.header("Text Comparison")
        
        # Generate comparison visualization
        comparison_plot = compare_texts(input_text, transformed_text)
        st.image(comparison_plot, use_column_width=True)
        
        # Add explanation
        st.markdown("""
        ### Key Differences
        
        The comparison above highlights how style transfer affects text characteristics:
        
        1. **Vocabulary**: The classic style typically uses more unique words and archaic terms
        2. **Sentence Structure**: Classic text tends to have longer, more complex sentences
        3. **Word Choice**: Modern text uses contemporary terminology while classic text uses formal, literary language
        4. **References**: Modern text often includes data points and research, while classic text uses metaphors and philosophical references
        
        These differences reflect the distinct linguistic patterns of different historical periods and writing styles.
        """)

# Run the app
if __name__ == "__main__":
    main()
