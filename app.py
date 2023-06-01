from flask import Flask, jsonify, request
import data_processing

app = Flask(__name__)


@app.route('/api/player-stats', methods=['GET'])
def player_stats():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"error": 'userId query parameter is required'}), 400
    gameday_id = request.args.get('gamedayId')

    dfTeamOverall, dfPlayerOverall, dfShots = data_processing.get_stats(
        user_id, gameday_id)

    return jsonify({
        "dfShots": dfShots.to_dict(orient="records"),
        "dfPlayerOverall": dfPlayerOverall.to_dict(orient="records"),
        "dfTeamOverall": dfTeamOverall.to_dict(orient="records")
    })


# other routes as necessary


if __name__ == "__main__":
    app.run(debug=True)
