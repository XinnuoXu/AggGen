 <template>
   <div class="container is-fluid home">
     <div class="columns" :style="{ display: display.landing }">
            <div class="column is-8 is-offset-2 box content">
                <component :is="dynamicLanding"></component>
                <div align="center" style="margin-bottom: 2rem">
                    <button class="button is-primary is-large"
                    v-on:click="closeLanding()">I consent</button>
                </div>
            </div>
     </div>

     <div class="columns" :style="{ display: display.content }">
       <div class="column">
         <div class="content" align="center">
             <h2>Please don't refresh the page.</h2>
         </div>
         <div class="box document">
           <a class="notice"> <b>Annotation Guideline</b></a>
           <blockquote style="font-size:18px">
            <li>Each triple must be selected <strong>once and only once</strong> in the <strong>entire questionnaire</strong>.</li>
            <li>In each section, <strong>all triples</strong> that are verbalized by the highlight need to be selected. </li>
            <li>Each section must have <strong>at least one</strong> selected triple. </li>
            </blockquote>
          </div>    
         <div class="box document">
           <Document v-on:ClickCheckbox="updateSelectionSummary"
                     v-on:noDocument="showMessage(
                     '<h1>The server is busy! Please wait 15 seconds and press refresh!</h1>')"
                     v-on:annotationDone="sendResult"
                     v-on:gotResult="saveDocStatusId"
                     :project_id="project_id"
                     :maxTokens="maxTokens"
                     :maxHL="maxHL"></Document>
         </div> 
       </div> 

        <div class="column is-4" :style="{ display: display.content }">
            <div class="box summary">
              <div class="content">
                <h2>Selected Triples</h2>
                <span v-bind:key="pd" v-for="(pd, index) in paired_data">
                  <h4 class="my-title">Triples in highlighted section {{index+1}}</h4>
                    <span v-bind:key="s" v-for="s in pd.Selection"> {{s}}<br/></span>
                  <hr>
                </span>
                <h2>Remaining Triples</h2>
                <span v-bind:key="k" v-for="k in left_triples"> {{k}}<br/></span>
              </div>
            </div>
          </div>
          
      </div>

     <div class="columns" :style="{ display: display.message }">
        <div class="column is-8 is-offset-2 box content">
            <div align="center" v-html="message">
            </div>
        </div>
     </div>
     
   </div>
</template>

<script>
/* eslint no-unused-vars: ["error", { "args": "none" }] */
/* eslint no-continue: "off" */
import Document from '@/components/Annotator/Document.vue';
import LandingHighlightMturk from '@/components/LandingMTurk/LandingHighlight.vue';
import vueSlider from 'vue-slider-component';
import 'vue-slider-component/theme/antd.css'

// const randomColor = require('randomcolor');
const axios = require('axios');

export default {
  name: 'Annotation',
  components: {
    LandingHighlightMturk,
    Document,
    vueSlider
  },
  data() {
    return {
      project_id: this.$route.params.project_id,
      result_id: '',
      is_mturk: this.$route.params.mturk,
      display: {
        content: 'none',
        landing: 'block',
        message: 'none',
        test: 'none',
      },
      resultJSON: {},
      message: '',
      paired_data: [],
      left_triples: [],
    };
  },
  computed: {
    dynamicLanding() {
      return 'LandingHighlightMturk';
    },
    mTurkDisplay() {
      return 'none';
    },
  },
  methods: {
    saveDocStatusId(arg) {
      this.ex_status_id = arg.ex_status_id;
    },

    updateSelectionSummary(arg) {
      this.paired_data = arg.paired_data;
      this.left_triples = arg.left_triples;
    },

    sendResult(arg) {
      this.resultJSON = arg.resultJSON;
      this.turkCode = arg.turkCode;
      this.resultJSON.result_id = this.result_id;
      this.resultJSON.mturk_code = this.turkCode;
      
      axios.post('project/save_result/annotation', this.resultJSON)
        .then(() => {
          this.$toast.open({
            message: 'Submission successful.',
            type: 'is-success',
          });
          let text = '<p>Please enter this code:</p>' + `<blockquote>${this.turkCode}</blockquote>`;
          this.showMessage(`<h3>Thank you for submitting!</h3><br/> ${text}`);
        })
        .catch((error) => {
          this.$toast.open({
            message: `${error}`,
            type: 'is-danger',
          });
        });
    },

    closeLanding() {
      this.display.content = 'flex';
      this.display.landing = 'none';
      window.scrollTo(0, 0);
      axios.get(`result/annotation/${this.ex_status_id}`)
        .then((response) => {
          this.result_id = response.data.result_id;
        });
        
    },

    showMessage(message) {
      this.display.landing = 'none';
      this.display.content = 'none';
      this.display.message = 'flex';
      this.display.test = 'none';
      this.message = message;
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.document {
  font-family: 'Lora', serif;
  font-size: 1.2rem;
  line-height: 1.5rem;
}
.my-title {
    font-size: 1rem;
}
.my-text {
    font-size: 0.9rem;
}
.summary {
    position: sticky;
    position: -webkit-sticky;
    top: 70px;
}
.instruction {
    position: sticky;
    position: -webkit-sticky;
    top: 70px;
}
.home {
  padding-top: 25px;
}
.notice {
        background-color: red;
        color: yellow;
}
</style>
