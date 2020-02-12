// app to text
const a2t_module = (function () {

    const MODE_PRED_BUTTONS = ["btn_show_combined", "btn_show_ids",
        "btn_show_strings", "btn_show_methods"];
    const MODES = ["combined", "strings", "ids", "methods", "actual"];

    const TFIDF_TOP5_SCALE = 1.59 * 0.75;

    let num_words = 10;
    let loaded = {};
    let currentFile = null;

    // ----------------------------------

    function show() {
        $('#panel_a2t').show();
        initButtons();
    }

    function hide() {
        $('#panel_a2t').hide();
    }

    function load(file, mode, callback) {
        if (currentFile === file) {
            return callback(true);
        }
        currentFile = file;

        $.getJSON(file).done(loaded_data => {
            loaded = loaded_data['a2t'] || {};
            callback(true);
            initHeader(loaded_data);
            $('#text_actual').hide();
            $('#text_actual').find('.card-body').html(loaded.text_actual || "")
            showCloud(mode);
        }).fail(() => callback(false));
    }

    function initHeader(header_data) {
        if (header_data['app_icon'] && header_data['app_icon'].length > 0) {
            $('#app_icon').attr('src', header_data['app_icon']);
            $('#app_icon_col').show();
        } else {
            $('#app_icon_col').hide();
        }

        const app_name = header_data['app_name'] && header_data['app_name'].length > 0 ? header_data['app_name'] : "Unknown App";
        $('#app_name').text(app_name);

        if (header_data['package']) {
            const link = "https://play.google.com/store/apps/details?hl=en_US&id=" + header_data['package']
            $('#store_link').attr('href', link).show();
        } else {
            $('#store_link').hide();
        }

        //$('#dangerous_permissions').text(loaded['dangerous_permissions'] || "?");
        $('#package').text(header_data['package']);
        $('#version').text(header_data['version']);
    }

    function initButtons() {
        for (let i = 0; i < MODE_PRED_BUTTONS.length; i++) {
            const btn_id = MODE_PRED_BUTTONS[i];
            const btn = $('#' + btn_id);
            btn.prop("onclick", null).off("click");
            btn.click(e => {
                const mode = btn_id.split("_")[2];
                showCloud(mode);
            })
        }
        const btn_actual = $('input[name=pred_actual]');
        btn_actual.prop("onchange", null).off("change");
        btn_actual.change(function () {
            const mode = $(this).val() === 'actual' ? 'actual' : 'combined';
            showCloud(mode);
        });
        $('label').click(e => $(e.target).find('input').click());
        $('input[name=pred_actual]')[0].click();
    }

    function showCloud(mode) {
        console.log("show cloud", mode);

        $('#input_influences').hide();
        //$('#show_cloud_container').hide();
        $('#cloud_description').show();

        $('#cloud_container').html("");
        $('#cloud_container').css({opacity: 1.0});
        $('#cloud_container').jQCloud("destroy");
        $('#prediction_quality').hide();

        if (MODES.indexOf(mode) === -1) {
            console.error("mode not available");
            mode = "ids";
        }

        $('#mode li').removeClass('active');
        $('#btn_show_' + mode).parents('li').addClass('active');

        let words = {};

        let quality = 100;
        $('#text_actual').hide();
        if (mode === 'combined') {
            $('ul#mode').show();
            for (let m in loaded.words_pred) {
                for (let w in loaded.words_pred[m]) {
                    words[w] = (words[w] || 0.) + loaded.words_pred[m][w]
                }
            }
            quality = showPredictionQualityProgressBar(words);
        } else if (mode === 'actual') {
            $('ul#mode').hide();
            words = loaded.words_actual || {'None': 1};
            $('#text_actual').show();
        } else {
            $('ul#mode').show();
            words = loaded.words_pred[mode];
            quality = showPredictionQualityProgressBar(words);
        }

        for (let w in words) {
            if (w.indexOf(" ") !== -1) {
                const ngram_tokens = w.split(" ");
                const combined_weight = ngram_tokens.reduce((pv, w) => pv + words[w], 0);

                words[w] = Math.max(combined_weight, words[w]);
                for(let i=0; i<ngram_tokens.length; i++) {
                    words[ngram_tokens[i]] = 0
                }
            }
        }

        let words_output = [];
        for (let w in words) {
            if(words[w] === 0) continue;
            words_output.push({text: w, weight: words[w], handlers: {'click': e => showInfluences(mode, e)}});
        }

        words_output = words_output.sort((a, b) => b.weight - a.weight);
        words_output.length = Math.min(num_words, words_output.length);

        //if (quality < 35) {
        //    $('#cloud_container').css({opacity: 0.4});
        //}

        words_output.forEach(w => {
            console.log(w.weight, w.text);
        });

        $('#cloud_container').jQCloud(words_output);
    }

    function setNumWords(num_words_new) {
        num_words = num_words_new;
    }

    function showInfluences(mode, e) {
        if(!loaded.input_values[mode]) {
            main.toast("Please select any mode other than 'Combined' first.")
            return;
        }

        const el_input_influences = $('#input_influences');
        const el_cloud_container = $('#cloud_description');
        el_input_influences.show();
        el_cloud_container.hide();
        el_input_influences.removeClass('jqcloud');
        el_input_influences.css({height: 'auto', width: 'auto'});

        el_input_influences.off('click').on('click', () => {
            el_input_influences.hide();
            el_cloud_container.show();
        });

        const output_word = e.target.textContent;
        el_input_influences.html("");
        const cloud_data = loaded.input_values[mode][output_word].map(influence => ({text: influence[0], weight: influence[1]}))

        if(mode === 'methods') {
            el_input_influences.append($('<h6>').text("Influences for whole method-based description"));
            cloud_data.length = Math.min(cloud_data.length, 100);
            cloud_data.forEach(inf => {
                const el_p = $('.template_influence_method').first().clone();
                const weight_str = (""+inf.weight).substr(1, 7);
                el_p.find('.inf_heat').text(weight_str);
                const parts = inf.text.split(";->");
                el_p.find('.inf_class').text(parts[0].substr(1).split("/").join("."));
                el_p.find('.inf_method').text(parts[1]);
                el_input_influences.append(el_p);
            });
        }
        else {
            el_input_influences.append($('<h6>').text("Influences for '"+output_word+"'"));
            cloud_data.length = Math.min(cloud_data.length, 20);
            el_input_influences.jQCloud(cloud_data);
        }
    }

    function showPredictionQualityProgressBar(words) {
        const top_word_heat_values = Object.values(words || {}).sort();
        if(top_word_heat_values.length == 0) return 0
        top_word_heat_values.reverse();
        top_word_heat_values.length = 5;
        const quality = Math.min((top_word_heat_values.reduce((pv, cv) => pv + cv)) / TFIDF_TOP5_SCALE * 100, 100);
        $('#prediction_quality').find('.progress').html('');
        const div_progressbar = $('<div class="progress-bar">');
        div_progressbar.attr('aria-valuenow', quality);
        div_progressbar.attr('aria-valuemin', 0);
        div_progressbar.attr('aria-valuemax', 100);
        div_progressbar.css({'width': '' + quality + '%'});
        const bg_class = (quality > 80) ? 'bg-success' : (quality < 30) ? 'bg-danger' : 'bg-warning';
        div_progressbar.addClass(bg_class);
        $('#prediction_quality').find('.progress').append(div_progressbar);
        $('#prediction_quality').show();
        return quality;
    }

    return {
        load: load,
        show: show,
        hide: hide,
        setNumWords: setNumWords
    }

});

// main module
const main = (function () {

    let _list = [];

    const a2t = a2t_module();

    let active_file = null;
    let _toast_timeout = null;

    function onDocumentLoad() {
        if(window.location.hash.indexOf("a2t") > -1) {
            loadFilesList(() => {
                const params = {};
                document.location.hash.split(",").forEach(keyvalue => {
                    const parts = keyvalue.split("=");
                    params[parts[0]] = parts.length === 2 ? parts[1] : 0;
                });
                loadForPrintA2t(params.file, params.mode);
                if(params.numwords > 0) {
                    a2t.setNumWords(params.numwords);
                }
            })
        }
        else {
            loadFilesList();
            $("#menu-toggle").click((e) => {
                e.preventDefault();
                $("#wrapper").toggleClass("toggled");
            });
            toast('Loaded random app. Use the menu to see other results.');
        }

    }

    function loadForPrintA2t(file, mode) {
        console.log("load for print ", file, mode);
        activateAndLoad(file, mode,() => {
            window.setTimeout(() => {
                $('body').html($('#cloud_container').clone());
                $('body').removeClass('bg-light');
                console.log('loaded for print');
            }, 1000);
        });
    }

    function loadFilesList(callback) {
        $.getJSON('list.json', function (obj) {
            _list = obj || [];

            if (!_list || !_list.length) {
                error();
                return
            }

            _list.sort((a, b) => {
                a = a.title;
                b = b.title;
                if (a.toLowerCase() < b.toLowerCase()) return -1;
                if (a.toLowerCase() > b.toLowerCase()) return 1;
                return 0;
            });

            const random_app = _list[parseInt(Math.random() * _list.length)];
            activateAndLoad(random_app.file, null, callback);

            $('#apps_list').html("");
            _list.forEach(element => {
                const el = $('#template_app_list').clone();
                el.find(".title").text(element.title);
                if (element.a2t) {
                    el.find(".btn_description").click(() => {
                        activateAndLoad(element.file, null);
                    });
                }
                el.css({display: 'block'});
                $('#apps_list').append(el);
            })
        });
    }

    function error() {
        $('#main_content_container').hide();
        $('#main_loading').html('<div class="alert alert-danger" role="alert">\n' +
            '  <h4 class="alert-heading">Error occured</h4>\n' +
            '  <p>The given app result files seem to be invalid.</p>\n' +
            '</div>');
    }

    function showMain() {
        $('#main_content_container').show();
        $('#main_loading').hide();
    }

    function activateAndLoad(file, mode, callback) {
        $('#main_loading').show();
        $('#main_content_container').hide();
        active_file = file;

        const callbackHandler = success => {
            handleFileLoadResult(success);
            if(callback) callback(success);
        };

        a2t.show();
        a2t.load('apps/' + file, mode, callbackHandler);

        if ($('#apps_list').is(':visible')) {
            $('button.navbar-toggler').click();
        }

    }

    function handleFileLoadResult(success) {
        if (!success) {
            error();
        } else {
            showMain();
        }
    }

    $(document).ready(() => {
        window.setTimeout(onDocumentLoad, 800);

        $(document).keyup(function (e) {
            if (e.which !== 37 && e.which !== 39) return;

            const direction = (e.which === 37) ? -1 : 1;
            const idx_current = _list.map(f => f.file).indexOf(active_file);
            const idx_new = idx_current + direction;

            if (idx_new >= 0 && idx_new < _list.length) {
                activateAndLoad(_list[idx_new].file, null);
                toast('App ' + (idx_new + 1) + ' of ' + _list.length);
            } else {
                toast('No more apps');
            }
        });
    });

    function toast(text) {
        const minitoast = $('#minitoast');
        minitoast.find('#minitoast_text').text(text);
        minitoast.fadeIn();
        clearTimeout(_toast_timeout);
        _toast_timeout = setTimeout(() => minitoast.fadeOut(), 4500);
    }

    return {
        toast: toast
    }

})();
