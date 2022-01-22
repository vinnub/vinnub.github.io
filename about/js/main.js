/* =================================
------------------------------------
	Civic - CV Resume
	Version: 1.0
 ------------------------------------ 
 ====================================*/



'use strict';


$(window).on('load', function() { 
	/*------------------
		Preloder
	--------------------*/
	$(".loader").fadeOut(); 
	$("#preloder").delay(400).fadeOut("slow");

});


(function($) {

	/*------------------
		Background set
	--------------------*/
	$('.set-bg').each(function() {
		var bg = $(this).data('setbg');
		$(this).css('background-image', 'url(' + bg + ')');
	});


	$('.review-slider').owlCarousel({
		loop: true,
		nav: false,
		dots: true,
		items: 1,
		autoplay: true
	});



	$('.progress-bar-style').each(function() {
		var progress = $(this).data("progress");
		var prog_width = progress + '%';
		if (progress <= 100) {
			$(this).append('<div class="bar-inner" style="width:' + prog_width + '"><span>' + prog_width + '</span></div>');
		}
		else {
			$(this).append('<div class="bar-inner" style="width:100%"><span>' + prog_width + '</span></div>');
		}
	});


	$('.lan-prog').each(function() {
		var progress = $(this).data("lanprogesss");
		var ele      = '<span></span>';
		var ele_fade = '<span class="fade-ele"></span>';
		
		for (var i = 1; i <= 5; i++) {
			if(i <= progress){
				$(this).append(ele);
			} else {
				$(this).append(ele_fade);
			}
		}
	});


	/*------------------
		Popup
	--------------------*/
	$('.portfolio-item .port-pic').magnificPopup({
		type: 'image',
		mainClass: 'img-popup-warp',
		removalDelay: 500,
	});




if($().circleProgress){

	//Set progress circle 1
	$("#progress_python").circleProgress({
		value: 0.90,
		size: 125,
		thickness: 8,
		fill: "#426280",
		emptyFill: "rgba(0, 0, 0, 0)"
	});
	//Set progress circle 2
	$("#progress_R").circleProgress({
		value: 0.85,
		size: 125,
		thickness: 8,
		fill: "#426280",
		emptyFill: "rgba(0, 0, 0, 0)"
	});

	//Set progress circle white
	$("#progress_MATLAB").circleProgress({
		value: 0.75,
		size: 125,
		thickness: 8,
		fill: "#426280",
		emptyFill: "rgba(0, 0, 0, 0)"
	});

	//Set progress circle white
	$("#progress_gephi").circleProgress({
		value: 0.650,
		size: 125,
		thickness: 8,
		fill: "#426280",
		emptyFill: "rgba(0, 0, 0, 0)"
	});

	//Set progress circle 1
	$("#progress_html").circleProgress({
		value: 0.50,
		size: 125,
		thickness: 8,
		fill: "#426280",
		emptyFill: "rgba(0, 0, 0, 0)"
	});

	//Set progress circle skyblue
	$("#progress5").circleProgress({
		value: 0.50,
		size: 125,
		thickness: 8,
		fill: "#009fff",
		emptyFill: "rgba(0, 0, 0, 0)"
	});

	//Set progress circle skyblue
	$("#progress6").circleProgress({
		value: 0.83,
		size: 125,
		thickness: 2,
		fill: "#009fff",
		emptyFill: "rgba(0, 0, 0, 0)"
	});
}

})(jQuery);

