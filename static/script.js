function handleMove(source, target, piece, newPos, oldPos, orientation) {
    var move = source + target;
    $.ajax({
        type: "POST",
        url: "/make_move",
        contentType: "application/json",
        data: JSON.stringify({ move: move }),
        success: function (data) {
            if (data.error) {
                alert(data.error);
                board.position(oldPos);
            } else {
                board.position(data.board);
                if (data.game_over) {
                    $("#message").text("Game Over! " + data.result);
                }
                else {
                    $("#progress-container").show();
                    $("#message").text("AI is thinking...");
                    aiMove();
                }
            }
        },
        error: function (error) {
            console.log(error);
        }
    });
}

function aiMove() {
    $.ajax({
        type: "GET",
        url: "/ai_move",
        success: function (data) {
            if (data.error) {
                alert(data.error);
            } else if (data.game_over) {
                board.position(data.board);
                $("#message").text("Game Over! " + data.result);
                $("#progress-container").hide();
            } else if (data.ai_move) {
                board.position(data.board);
                $("#message").text("AI played: " + data.ai_move);
                $("#progress-container").hide();
            } else {
                updateProgress(data.progress, data.total);
                aiMove();
            }
        },
        error: function (error) {
            console.log(error);
        }
    });
}

function updateProgress(progress, total) {
    var percentage = (progress / total) * 100;
    $("#progress-bar").css("width", percentage + "%");
}