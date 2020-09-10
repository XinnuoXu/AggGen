<template>
    <div>
          <span v-bind:key="pd" v-for="(pd, index) in paired_data">
            <div>
                <span>&#127807; <b>Highlight {{index+1}} in the text: </b></span>
                <p v-html="pd.Sentence"></p><br>
                <span>&#127793; <b>Candidate triples: </b></span><br>
                <span v-bind:key="s" v-for="s in pd.Input">
                    <input type="checkbox" :value="s.Relation" v-model="pd.Selection" :id="s.ID" v-on:click="CheckSelection(s.Relation, s.ID)"> 
                      <span class="checkbox-label"> {{s.Record}} </span> <br>
                </span>
                <hr>
            </div>
          </span>

        <div align="center">
            <button class="button is-primary" :disabled="disable_button"
                    v-on:click="saveAnnotation()">{{ disable_status }}</button>
        </div>

    </div>

</template>

<script>

const axios = require('axios');

function getFile() {
  axios.get(`project/annotation/highlight/${this.project_id}/single_doc`)
    .then((response) => {
      this.ex_status_id = response.data.ex_status_id;
      this.turkCode = response.data.turk_code;
      this.sanity_check = JSON.parse(response.data.sanity_check).text;
      this.$emit('gotResult', {
        ex_status_id: this.ex_status_id,
      })
      this.paired_data = JSON.parse(response.data.paired_data)
      this.$emit('ClickCheckbox', {
        paired_data: this.paired_data,
        left_triples: this.left_triples,
      })
      
      //console.log(`response.data.summary ${response.data.tgt_json}`)
      console.log(`Loaded doc doc_id: ${response.data.ex_id}`)
      console.log(`Loaded doc doc_id PARSED: ${JSON.parse(response.data.ex_id)}`)
      console.log(response.data)
    })
    .catch(() => {
      this.$emit('noDocument');
    });

}

export default {
  name: 'Document',
  props: ['project_id', 'maxTokens', 'maxHL'],
  data() {
    return {
      ex_status_id: '',

      disable_button: true,

      turkCode: '',
      src: '',
      tgt: '',
      sanity_check: '',
      
      selected_triples: [],
      left_triples: [],
      paired_data: [],
    };
  },
  computed: {
    disable_status() {
      console.log(this.left_triples.length );
      console.log(this.selected_triples.length);
      this.disable_button = (this.left_triples.length > 0 || this.selected_triples.length == 0);
      if (this.disable_button) {
        return `${this.left_triples.length} records to label`;
      }
      return 'Click to submit';
    },
  },
  methods: {
    saveAnnotation() {
      const resultJSON = {
        project_id: this.project_id,
        status_id: this.ex_status_id,
        result_json: this.paired_data,
      };
      this.$emit('annotationDone', {
        resultJSON,
        answer: this.sanity_answer,
        turkCode: this.turkCode,
      });
    },

    CheckSelection(clickrelation, cid) {
      this.selected_triples = [];
      for (var i = 0; i < this.paired_data.length; ) {
        for (var j = 0; j < this.paired_data[i].Selection.length; ) {
          this.selected_triples.push(this.paired_data[i].Selection[j]);
          j++;
        }
        i++;
      }
  
      this.left_triples.length = 0;
      for (var k = 0; k < this.paired_data[0].Input.length; ) {
        var r = this.paired_data[0].Input[k].Relation;
        if (clickrelation == r){
          if (!document.getElementById(cid).checked){
            this.left_triples.push(r);
          }
        }
        else{
          if (!this.selected_triples.includes(r)) {
            this.left_triples.push(r);
          }
        }
        k++;
      } 

      if (document.getElementById(cid).checked && this.selected_triples.includes(clickrelation)){
        alert('This triple has been selected. Please uncheck the previous one to select this one.');
        document.getElementById(cid).checked = false;
      }

    },
  },
  mounted: function onMounted() {
    getFile.call(this);
  },
};

</script>

<style>
.highlight{
  background-color: greenyellow;
  color: black;
}
</style>
