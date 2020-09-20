<template>
    <div class="document">
        <!-- eslint-disable -->
        <h3 align="center">
            Task Description: Information alignment
        </h3>
        In this task, you need to align some triples to a highlight in a text, which is the verbalisation of these triples. Please finish reading the example below and click 'I consent' to start your task.
        <hr/>
        In each task, you will be given a <strong>text</strong> and a list of <strong>triples</strong>
        (A triple consists of a <em>subject</em>, an <em>object</em> and the <em>relation</em> in between them. 
        They are written in format <u>Subject [Relation] Object</u>.).<br/><br/>
        The task, is devided into a sequence of <strong>Highlight Sections</strong>.
        In each highlight section, a part of the text will be <a class="highlight">highlighted</a>, and you need to tick triples that, you believe, are verbalized by the highlighted text.
        <br/><br/>
        For example, the <strong>text</strong> is:
        <blockquote style="font-size:18px">
            The Bakewell tart, containing fruit preserves, is a variation of Bakewell pudding.
        </blockquote>
        and the <strong>triples</strong> are:
        <blockquote style="font-size:18px">
            <li>Bakewell pudding [<em>dishVariation</em>] Bakewell tart</li>
            <li>Bakewell tart [<em>ingredient</em>] Fruit preserves</li>
        </blockquote>
        <!-- 
        The highlight in text:
        <blockquote style="font-size:18px">
            <mark>The Bakewell tart, containing fruit preserves,</mark> is a variation of Bakewell pudding.
        </blockquote>
        should be aligned with the second triple:
        <blockquote style="font-size:18px">
            <li>Bakewell tart [<em>ingredient</em>] Fruit preserves</li>
        </blockquote>
        since the highlight describes the second triple in natural language. Out of the same reason, the highlight:
        <blockquote style="font-size:18px">
            The Bakewell tart, containing fruit preserves, <mark>is a variation of Bakewell pudding. </mark>
        </blockquote>
        should be aligned with the first triple:
        <blockquote style="font-size:18px">
            <li>Bakewell pudding [<em>dishVariation</em>] Bakewell tart</li>
        </blockquote> 
        <strong>Your task</strong> is to do the alignments by finishing a questionnaire. -->
        
        After you click the 'I consent' button, you will see a <strong>questionnaire</strong>: <br/><br/>       
        <div class="box document">
            <div>
                &#127807; <b>Highlight Section 1: </b><br/>
                <a class="highlight">The Bakewell tart , containing fruit preserves,</a> is a variation of Bakewell pudding.<br/><br/>
                &#127793; <b>Candidate triples: </b><br/>
                <input type="checkbox" >Bakewell pudding [dishVariation] Bakewell tart <br>
                <input type="checkbox" checked >Bakewell tart [ingredient] Fruit preserves
                <hr>
            </div>
            <div>
                &#127807; <b>Highlight Section 2: </b><br/>
                The Bakewell tart , containing fruit preserves, <a class="highlight">is a variation of Bakewell pudding.</a><br/><br/>
                &#127793; <b>Candidate triples: </b><br/>
                <input type="checkbox" checked >Bakewell pudding [dishVariation] Bakewell tart <br>
                <input type="checkbox" >Bakewell tart [ingredient] Fruit preserves
                <hr>
            </div>
            <div>
                &#127800; <b>Now tick triples that are <mark>NOT</mark> mentioned in highlight sections above: </b><br/><br/>
                &#127793; <b>Candidate triples: </b><br/>
                <input type="checkbox" >Bakewell pudding [dishVariation] Bakewell tart <br>
                <input type="checkbox" >Bakewell tart [ingredient] Fruit preserves
            </div>
        </div>
        <!--It breaks down the highlights of the given text into serveral sections. Each section starts with a highlight, followed by its candidate triples. -->        
        <!--<strong>You need to</strong> go through each section and select the triples that, you believe, are verbalized by the highlight in the same section. -->
        

        <hr/>
        <a class="notice"> <b>Please Note </b></a>
        <ul>
            <li>Candidate triples in each section are all the <strong>same</strong>.</li>
            <li>If you believe that a triple has <strong>NOT</strong> been described in any highlight, please select them in the last section.</li>
            <li>Each triple must be selected <strong>once and only once</strong> in the <strong>entire questionnaire</strong>. The submission button will not be activated if there are unselected triples, or one triple is selected in more than one sections.</li>
            <li>In each section, all triples that are verbalized by the highlight need to be selected. <strong>For some sections</strong>, it may be possible that <strong>multiple, or even zero triples</strong> are the right answer. The following examples are all regarded as valid.</li>
            <li>There is a <strong>code</strong> that you <strong>have to</strong> copy and paste to Amazon platform at the end of the task.</li>
            <li>We have a validation check for each task. Only submissions that offer resonable annotations will be deemed as a valid HIT.</li>
        </ul>

        <div>
            <collapse :selected="false">
                <div slot="collapse-header">
                    <b>Click here to View Example 1</b>
                </div>
                <div slot="collapse-body">
                    <div class="box document">
                        <div>
                            &#127807; <b>Highlight Section 1: </b><br/>
                            <a class="highlight">200 Public Square completed in 1985</a> is in Cleveland (part of Cuyahoga County) in Ohio , U.S. .<br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" > 200 Public Square [location] Cleveland <br>
                            <input type="checkbox" checked >200 Public Square [completionDate] 1985 <br>
                            <input type="checkbox" >Cleveland [isPartOf] Cuyahoga County, Ohio <br>
                            <input type="checkbox" >Cleveland [isPartOf] Ohio <br>
                            <input type="checkbox" >Cleveland [country] United States
                            <hr>
                        </div>
                        <div>
                            &#127807; <b>Highlight Section 2: </b><br/>
                            200 Public Square completed in 1985 <a class="highlight"> is in Cleveland (part of Cuyahoga County) in Ohio , U.S. .</a><br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" checked> 200 Public Square [location] Cleveland <br>
                            <input type="checkbox" >200 Public Square [completionDate] 1985 <br>
                            <input type="checkbox" checked>Cleveland [isPartOf] Cuyahoga County, Ohio <br>
                            <input type="checkbox" checked>Cleveland [isPartOf] Ohio <br>
                            <input type="checkbox" checked>Cleveland [country] United States
                            <hr>
                        </div>
                        <div>
                            &#127800; <b>Now tick triples that are <mark>NOT</mark> mentioned in highlight sections above: </b><br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" > 200 Public Square [location] Cleveland <br>
                            <input type="checkbox" >200 Public Square [completionDate] 1985 <br>
                            <input type="checkbox" >Cleveland [isPartOf] Cuyahoga County, Ohio <br>
                            <input type="checkbox" >Cleveland [isPartOf] Ohio <br>
                            <input type="checkbox" >Cleveland [country] United States
                        </div>
                    </div>
                </div>
            </collapse>
        </div>

        <div>
            <collapse :selected="false">
                <div slot="collapse-header">
                    <b>Click here to View Example 2</b>
                </div>
                <div slot="collapse-body">
                    <div class="box document">
                        <div>
                            &#127807; <b>Highlight Section 1: </b><br/>
                            <a class="highlight">Adare Manor 's architect was Philip Charles Hardwick .</a> it was completed in 1862 .<br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" checked> Adare Manor [architect] Philip Charles Hardwick <br>
                            <input type="checkbox" >Adare Manor [completionDate] 1862 <br>
                            <input type="checkbox" >Adare Manor [owner] J. P. McManus
                            <hr>
                        </div>
                        <div>
                            &#127807; <b>Highlight Section 2: </b><br/>
                            Adare Manor 's architect was Philip Charles Hardwick .<a class="highlight"> it was completed in 1862 .</a> <br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" > Adare Manor [architect] Philip Charles Hardwick <br>
                            <input type="checkbox" checked>Adare Manor [completionDate] 1862 <br>
                            <input type="checkbox" >Adare Manor [owner] J. P. McManus
                            <hr>
                        </div>
                        <div>
                            &#127800; <b>Now tick triples that are <mark>NOT</mark> mentioned in highlight sections above: </b><br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" > Adare Manor [architect] Philip Charles Hardwick <br>
                            <input type="checkbox" >Adare Manor [completionDate] 1862 <br>
                            <input type="checkbox" checked>Adare Manor [owner] J. P. McManus
                            <hr>
                        </div>
                    </div>
                </div>
            </collapse>
        </div>

        <div>
            <collapse :selected="false">
                <div slot="collapse-header">
                    <b>Click here to View Example 3</b>
                </div>
                <div slot="collapse-body">
                    <div class="box document">
                        <div>
                            &#127807; <b>Highlight Section 1: </b><br/>
                            <a class="highlight">The United States ,</a> includes the African Americans ethnic group uses the English language and is the location of Albany , Georgia .<br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" >United States [language] English language <br>
                            <input type="checkbox" >Albany, Georgia [country] United States <br>
                            <input type="checkbox" >United States [ethnicGroup] African Americans
                            <hr>
                        </div>
                        <div>
                            &#127807; <b>Highlight Section 2: </b><br/>
                            The United States ,<a class="highlight">includes the African Americans ethnic group uses the English language </a>and is the location of Albany , Georgia .<br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" checked>United States [language] English language <br>
                            <input type="checkbox" >Albany, Georgia [country] United States <br>
                            <input type="checkbox" checked>United States [ethnicGroup] African Americans
                            <hr>
                        </div>
                        <div>
                            &#127807; <b>Highlight Section 3: </b><br/>
                            The United States ,includes the African Americans ethnic group uses the English language <a class="highlight">and is the location of Albany , Georgia .</a><br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" >United States [language] English language <br>
                            <input type="checkbox" checked>Albany, Georgia [country] United States <br>
                            <input type="checkbox" >United States [ethnicGroup] African Americans
                            <hr>
                        </div>
                        <div>
                            &#127800; <b>Now tick triples that are <mark>NOT</mark> mentioned in highlight sections above: </b><br/><br/>
                            &#127793; <b>Candidate triples: </b><br/>
                            <input type="checkbox" >United States [language] English language <br>
                            <input type="checkbox" >Albany, Georgia [country] United States <br>
                            <input type="checkbox" >United States [ethnicGroup] African Americans
                            <hr>
                        </div>
                    </div>
                </div>
            </collapse>
        </div>

        <hr>

        <p>Thank you for reading this</p>
        <p>
            Do you consent for all the information that you have submitted to be use anonymously for research purposes only?
        </p>
        <!-- eslint-enable -->
    </div>
</template>

<script>
    import Collapse from 'vue-collapse'
    export default {
        // name: 'LandingHighlightMTurk',
        name: "YourComponent",
        components: {
            Collapse
        }
    };
</script>

<style lang="scss" scoped>
    h3 {
        margin-top: 2rem;
        margin-bottom: 3rem;
    }

    .document {
        font-family: 'Lora', serif;
        font-size: 1.1rem;
        line-height: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }

    .highlight {
        background-color: greenyellow;
    }

    .notice {
        background-color: red;
        color: yellow;
    }
</style>

<style lang="scss">
    // Custom vue-collapse style
    .collapse .collapse-header {
        padding: 10px 10px 10px 40px;
    }

    .collapse .collapse-header:before {
        top: initial;
    }
</style>
